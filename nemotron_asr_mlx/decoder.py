"""RNNT decoder — PredictNetwork, JointNetwork, and greedy decode.

The predict network is a 2-layer LSTM that models the label history.
The joint network combines encoder and predict outputs to produce
logits over the vocabulary (1024 BPE tokens + 1 blank).

Greedy decode implements the standard RNNT loop: for each encoder frame,
repeatedly run predict->joint->argmax until blank or max_symbols.
"""

from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn
import numpy as np


BLANK_ID = 1024


# ------------------------------------------------------------------
# Multi-layer LSTM wrapper (batch_first, stacks per-layer states)
# ------------------------------------------------------------------

class MultiLayerLSTM(nn.Module):
    """Thin wrapper around multiple ``nn.LSTM`` layers.

    Accepts batch-first input ``[B, T, D]`` and an optional tuple of
    stacked hidden/cell states ``(h, c)`` each of shape ``[n_layers, B, H]``.
    Returns ``(output, (h, c))`` with the same conventions.
    """

    def __init__(self, input_size: int, hidden_size: int, num_layers: int = 1):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.layers = [
            nn.LSTM(input_size if i == 0 else hidden_size, hidden_size)
            for i in range(num_layers)
        ]

    def __call__(
        self,
        x: mx.array,
        h_c: tuple[mx.array, mx.array] | None = None,
    ) -> tuple[mx.array, tuple[mx.array, mx.array]]:
        # x: [B, T, D] -> transpose to [T, B, D] for nn.LSTM
        x = mx.transpose(x, (1, 0, 2))

        if h_c is None:
            h_list = [None] * self.num_layers
            c_list = [None] * self.num_layers
        else:
            h_all, c_all = h_c
            h_list = [h_all[i] for i in range(self.num_layers)]
            c_list = [c_all[i] for i in range(self.num_layers)]

        next_h = []
        next_c = []
        out = x
        for i, layer in enumerate(self.layers):
            all_h, all_c = layer(out, hidden=h_list[i], cell=c_list[i])
            out = all_h
            next_h.append(all_h[-1])   # last time-step hidden
            next_c.append(all_c[-1])   # last time-step cell

        # back to batch-first: [T, B, H] -> [B, T, H]
        out = mx.transpose(out, (1, 0, 2))
        return out, (mx.stack(next_h, axis=0), mx.stack(next_c, axis=0))


# ------------------------------------------------------------------
# PredictNetwork (label predictor)
# ------------------------------------------------------------------

class PredictNetwork(nn.Module):
    """RNNT prediction network — embedding + multi-layer LSTM + linear.

    Parameters
    ----------
    vocab_size : int
        Number of BPE tokens (excluding blank). Embedding has vocab_size+1
        entries so that the blank id can be used as the initial token.
    embed_dim : int
        Embedding and LSTM hidden dimension (640 for Nemotron).
    num_layers : int
        Number of stacked LSTM layers (2 for Nemotron).
    """

    def __init__(
        self,
        vocab_size: int = 1024,
        embed_dim: int = 640,
        num_layers: int = 2,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.embed = nn.Embedding(vocab_size + 1, embed_dim)  # +1 for blank-as-pad
        self.dec_rnn = MultiLayerLSTM(embed_dim, embed_dim, num_layers)

    def __call__(
        self,
        targets: mx.array | None,
        hidden: tuple[mx.array, mx.array] | None = None,
    ) -> tuple[mx.array, tuple[mx.array, mx.array]]:
        """
        Parameters
        ----------
        targets : mx.array | None
            Token ids, shape ``[B, T]``.  If ``None``, a single zero-frame
            is used (start-of-sequence).
        hidden : tuple | None
            ``(h, c)`` each ``[n_layers, B, H]``.

        Returns
        -------
        output : mx.array  ``[B, T, embed_dim]``
        new_hidden : tuple  ``(h, c)``
        """
        if targets is not None:
            x = self.embed(targets)
        else:
            batch = 1 if hidden is None else hidden[0].shape[1]
            x = mx.zeros((batch, 1, self.embed_dim))

        out, new_hidden = self.dec_rnn(x, hidden)
        return out, new_hidden


# ------------------------------------------------------------------
# JointNetwork
# ------------------------------------------------------------------

class JointNetwork(nn.Module):
    """RNNT joint network — projects encoder + decoder, adds, ReLU, output.

    Parameters
    ----------
    encoder_dim : int
        Encoder output dimension (1024 for Nemotron).
    decoder_dim : int
        Predict-network output dimension (640 for Nemotron).
    joint_dim : int
        Hidden dimension of the joint space (640 for Nemotron).
    vocab_size : int
        Number of BPE tokens (excluding blank). Output has vocab_size+1 logits.
    """

    def __init__(
        self,
        encoder_dim: int = 1024,
        decoder_dim: int = 640,
        joint_dim: int = 640,
        vocab_size: int = 1024,
    ):
        super().__init__()
        self.encoder_proj = nn.Linear(encoder_dim, joint_dim)
        self.decoder_proj = nn.Linear(decoder_dim, joint_dim)
        self.output = nn.Linear(joint_dim, vocab_size + 1)

    def __call__(self, encoder_out: mx.array, decoder_out: mx.array) -> mx.array:
        """
        Parameters
        ----------
        encoder_out : mx.array  ``[B, T, encoder_dim]``
        decoder_out : mx.array  ``[B, U, decoder_dim]``

        Returns
        -------
        logits : mx.array  ``[B, T, U, vocab_size+1]``
        """
        enc = self.encoder_proj(encoder_out)   # [B, T, J]
        dec = self.decoder_proj(decoder_out)   # [B, U, J]
        # Broadcast add: [B, T, 1, J] + [B, 1, U, J] -> [B, T, U, J]
        x = mx.expand_dims(enc, 2) + mx.expand_dims(dec, 1)
        x = nn.relu(x)
        return self.output(x)


# ------------------------------------------------------------------
# Greedy RNNT decode (streaming-compatible)
# ------------------------------------------------------------------

def greedy_decode(
    encoder_output: mx.array,
    predict_net: PredictNetwork,
    joint_net: JointNetwork,
    hidden: tuple[mx.array, mx.array] | None = None,
    last_token: int = BLANK_ID,
    max_symbols: int = 10,
) -> tuple[list[int], tuple[mx.array, mx.array] | None, int]:
    """Greedy RNNT decoding over encoder frames.

    Optimized version with:
    1. PredictNetwork caching: only re-run if token/hidden changes.
    2. Blank skipping: batch process frames until a non-blank is emitted.
    """
    T = encoder_output.shape[1]
    if T == 0:
        return [], hidden, last_token

    tokens: list[int] = []

    # Cache for PredictNetwork output
    cached_pred_out = None
    cached_new_h = None
    cached_token = None
    cached_h_in = None

    t = 0
    BLOCK_SIZE = 32
    while t < T:
        # 1. Update PredictNetwork cache if needed
        if (
            cached_pred_out is None
            or last_token != cached_token
            or hidden is not cached_h_in
        ):
            token_input = mx.array([[last_token]])
            cached_pred_out, cached_new_h = predict_net(token_input, hidden)
            cached_token = last_token
            cached_h_in = hidden

        # 2. Try blank-skipping for a block of frames
        # We process a small block to keep the compute bounded and avoid O(T^2)
        # re-computation if non-blanks are frequent.
        block_end = min(t + BLOCK_SIZE, T)
        block_frames = encoder_output[:, t:block_end, :]
        
        logits = joint_net(block_frames, cached_pred_out)
        preds = mx.argmax(logits[0, :, 0, :], axis=-1)
        preds_np = np.array(preds)
        
        non_blank_indices = np.where(preds_np != BLANK_ID)[0]
        
        if len(non_blank_indices) == 0:
            # All frames in this block are blank
            t = block_end
            continue
        
        # Move t to the first non-blank frame in this block
        first_nb = non_blank_indices[0]
        t += int(first_nb)
        
        # 3. Process the non-blank frame sequentially
        frame = encoder_output[:, t : t + 1, :]
        symbols_emitted = 0
        
        while symbols_emitted < max_symbols:
            # Re-run predict if needed (if we just emitted a non-blank)
            if last_token != cached_token or hidden is not cached_h_in:
                token_input = mx.array([[last_token]])
                cached_pred_out, cached_new_h = predict_net(token_input, hidden)
                cached_token = last_token
                cached_h_in = hidden
                
            logits = joint_net(frame, cached_pred_out)
            pred_token = int(mx.argmax(logits[0, 0, 0]))
            
            if pred_token == BLANK_ID:
                break
                
            tokens.append(pred_token)
            hidden = cached_new_h
            last_token = pred_token
            symbols_emitted += 1
            
        t += 1

    return tokens, hidden, last_token


# ------------------------------------------------------------------
# Beam-search RNNT decode (mAES — Modified Adaptive Expansion Search)
# ------------------------------------------------------------------

def beam_search_decode(
    encoder_output: mx.array,
    predict_net: PredictNetwork,
    joint_net: JointNetwork,
    hidden: tuple[mx.array, mx.array] | None = None,
    last_token: int = BLANK_ID,
    max_symbols: int = 10,
    beam_size: int = 4,
    score_fn: callable | None = None,
) -> tuple[list[int], tuple[mx.array, mx.array] | None, int]:
    """Beam-search RNNT decoding over encoder frames.

    Implements Modified Adaptive Expansion Search (mAES) for RNNT.
    For each encoder frame, hypotheses are expanded with non-blank tokens
    until all active hypotheses emit blank or hit *max_symbols*.  The top
    *beam_size* hypotheses (by log-probability) are kept at each step.

    When ``beam_size=1`` the result is identical to ``greedy_decode``.

    Parameters
    ----------
    encoder_output : mx.array
        Encoder features for a single utterance, shape ``[1, T, D]``.
    predict_net : PredictNetwork
    joint_net : JointNetwork
    hidden : tuple | None
        LSTM ``(h, c)`` state from the previous chunk (or None for start).
    last_token : int
        The last non-blank token emitted (or BLANK_ID at start).
    max_symbols : int
        Maximum non-blank symbols to emit per encoder frame.
    beam_size : int
        Number of hypotheses to keep at each expansion step.
    score_fn : callable | None
        Optional external scoring callback.  Called as
        ``score_fn(tokens, new_token) -> float`` and the returned value is
        added to the hypothesis score for non-blank expansions.

    Returns
    -------
    tokens : list[int]
        Decoded non-blank token ids (from the best hypothesis).
    new_hidden : tuple
        Updated LSTM state for the best hypothesis.
    new_last_token : int
        Last emitted non-blank token (or unchanged if only blanks).
    """
    # beam_size=1 without external scoring is exactly greedy decoding.
    # Delegate to avoid subtle differences in blank/max_symbols handling.
    if beam_size <= 1 and score_fn is None:
        return greedy_decode(
            encoder_output, predict_net, joint_net,
            hidden=hidden, last_token=last_token, max_symbols=max_symbols,
        )

    T = encoder_output.shape[1]
    if T == 0:
        return [], hidden, last_token

    # Each hypothesis: (tokens, score, hidden_state, last_token)
    beam = [([], 0.0, hidden, last_token)]

    for t in range(T):
        frame = encoder_output[:, t : t + 1, :]  # [1, 1, D]

        # Hypotheses that have emitted blank for this frame (done expanding)
        finished = []
        # Hypotheses still eligible to emit non-blank tokens
        active = [(hyp_tokens, hyp_score, hyp_hidden, hyp_last, 0)
                  for hyp_tokens, hyp_score, hyp_hidden, hyp_last in beam]

        while active:
            next_active = []

            for hyp_tokens, hyp_score, hyp_hidden, hyp_last, sym_count in active:
                # If this hypothesis already emitted max_symbols for this
                # frame, force it to finish without an extra joint step
                # (matches greedy_decode which simply exits its while loop).
                if sym_count >= max_symbols:
                    finished.append((
                        hyp_tokens, hyp_score, hyp_hidden, hyp_last,
                    ))
                    continue

                # --- Predict step ---
                token_input = mx.array([[hyp_last]])
                pred_out, new_h = predict_net(token_input, hyp_hidden)

                # --- Joint step ---
                logits = joint_net(frame, pred_out)  # [1, 1, 1, vocab+1]
                mx.eval(logits)
                log_probs = logits[0, 0, 0] - mx.logsumexp(logits[0, 0, 0])
                mx.eval(log_probs)

                # Blank transition — hypothesis is done for this frame.
                # Keep the *pre-predict* hidden state: in RNNT, blank means
                # "advance encoder, keep decoder state unchanged" (matches
                # greedy_decode which does not update hidden on blank).
                blank_score = float(log_probs[BLANK_ID])
                finished.append((
                    hyp_tokens,
                    hyp_score + blank_score,
                    hyp_hidden,
                    hyp_last,
                ))

                # Non-blank expansions
                nb_probs = log_probs.tolist()
                candidates = []
                for tok_id, tok_lp in enumerate(nb_probs):
                    if tok_id == BLANK_ID:
                        continue
                    candidates.append((tok_lp, tok_id))

                # Sort descending by log-prob, take top beam_size
                candidates.sort(key=lambda c: c[0], reverse=True)
                candidates = candidates[:beam_size]

                for tok_lp, tok_id in candidates:
                    new_score = hyp_score + tok_lp
                    if score_fn is not None:
                        new_score += score_fn(hyp_tokens, tok_id)
                    next_active.append((
                        hyp_tokens + [tok_id],
                        new_score,
                        new_h,
                        tok_id,
                        sym_count + 1,
                    ))

            # Prune active set to beam_size
            if next_active:
                next_active.sort(key=lambda h: h[1], reverse=True)
                active = next_active[:beam_size]
            else:
                active = []

        # Merge finished hypotheses with identical token sequences.
        # Uses Viterbi-style max (standard in RNNT beam search).
        merged: dict[tuple, tuple] = {}
        for hyp_tokens, hyp_score, hyp_hidden, hyp_last in finished:
            key = tuple(hyp_tokens)
            if key not in merged or hyp_score > merged[key][1]:
                merged[key] = (hyp_tokens, hyp_score, hyp_hidden, hyp_last)

        # Keep top beam_size for next frame
        ranked = sorted(merged.values(), key=lambda h: h[1], reverse=True)
        beam = ranked[:beam_size]

    # Return the best hypothesis
    best_tokens, _best_score, best_hidden, best_last = beam[0]
    return best_tokens, best_hidden, best_last
