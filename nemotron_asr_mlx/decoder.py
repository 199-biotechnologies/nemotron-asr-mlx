"""RNNT decoder — PredictNetwork, JointNetwork, and greedy decode.

The predict network is a 2-layer LSTM that models the label history.
The joint network combines encoder and predict outputs to produce
logits over the vocabulary (1024 BPE tokens + 1 blank).

Greedy decode implements the standard RNNT loop: for each encoder frame,
repeatedly run predict->joint->argmax until blank or max_symbols.
"""

import mlx.core as mx
import mlx.nn as nn


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

    This function is streaming-compatible: it accepts an initial decoder
    hidden state and last token, and returns the updated state so that
    decoding can continue with the next chunk of encoder output.

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
        Maximum non-blank symbols to emit per encoder frame (prevents
        infinite loops on degenerate inputs).

    Returns
    -------
    tokens : list[int]
        Decoded non-blank token ids.
    new_hidden : tuple
        Updated LSTM state.
    new_last_token : int
        Last emitted non-blank token (or unchanged if only blanks).
    """
    T = encoder_output.shape[1]
    if T == 0:
        return [], hidden, last_token

    tokens: list[int] = []

    for t in range(T):
        frame = encoder_output[:, t : t + 1, :]  # [1, 1, D]
        symbols_emitted = 0

        while symbols_emitted < max_symbols:
            # Predict step
            token_input = mx.array([[last_token]])
            pred_out, new_h = predict_net(token_input, hidden)
            # pred_out: [1, 1, pred_dim]

            # Joint step
            logits = joint_net(frame, pred_out)  # [1, 1, 1, vocab+1]
            mx.eval(logits)
            pred_token = int(mx.argmax(logits[0, 0, 0]))

            if pred_token == BLANK_ID:
                break

            tokens.append(pred_token)
            hidden = new_h
            last_token = pred_token
            symbols_emitted += 1

    return tokens, hidden, last_token
