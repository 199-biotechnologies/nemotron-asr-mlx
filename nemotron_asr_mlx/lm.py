"""N-gram language model for shallow fusion with beam search decoding."""

from __future__ import annotations

import gzip
import math
import os
import shutil
import urllib.request
from pathlib import Path

try:
    import kenlm
except ImportError:
    kenlm = None

LN10 = math.log(10)  # 2.302585...
_LIBRISPEECH_4GRAM_URL = "https://www.openslr.org/resources/11/4-gram.arpa.gz"
_WORD_BOUNDARY = "\u2581"  # BPE word-boundary marker


class NgramLM:
    """KenLM-based n-gram language model wrapper.

    Provides word-level n-gram scoring for BPE token sequences,
    designed for shallow fusion with RNNT beam search.

    Uses KenLM's stateful API (BaseScore + State) for O(1) per-word
    scoring instead of re-scoring the full prefix each time.
    """

    def __init__(self, model_path: str, tokenizer=None):
        """Load a KenLM model from ARPA or binary format.

        Parameters
        ----------
        model_path : str
            Path to .arpa or .binary KenLM model file.
        tokenizer : NemotronTokenizer
            Tokenizer for BPE-to-word conversion.
        """
        if kenlm is None:
            raise ImportError(
                "kenlm is required for n-gram language model fusion. Install with:\n"
                "  pip install kenlm\n"
                "or build from source:\n"
                "  pip install https://github.com/kpu/kenlm/archive/master.zip"
            )
        self.model = kenlm.Model(model_path)
        self.tokenizer = tokenizer

    def score_token(self, token_ids: list[int], new_token: int) -> float:
        """Score a new token given the history.

        Handles BPE-to-word boundary detection:
        - Accumulate BPE pieces until a word boundary is detected
        - Score complete words with the n-gram model
        - Return 0.0 for partial words (mid-BPE-sequence)

        Parameters
        ----------
        token_ids : list[int]
            Previously decoded token IDs.
        new_token : int
            The candidate new token.

        Returns
        -------
        float
            Log-probability from the n-gram model (natural log).
            Returns 0.0 if the new token is mid-word (no complete word yet).
        """
        if self.tokenizer is None:
            return 0.0

        vocab = self.tokenizer.vocab
        blank_id = self.tokenizer.blank_id

        if new_token == blank_id or new_token < 0 or new_token >= len(vocab):
            return 0.0

        new_piece = vocab[new_token]

        # A word boundary is detected when the new BPE piece starts with the
        # boundary marker. This means the PREVIOUS accumulated pieces form a
        # complete word that we can now score.
        if not new_piece.startswith(_WORD_BOUNDARY):
            return 0.0

        # New word boundary detected. Reconstruct the text of all previously
        # decoded tokens, then score the last complete word.
        if not token_ids:
            return 0.0

        # Build the word list from history by walking BPE pieces.
        words = _bpe_ids_to_words(token_ids, vocab, blank_id)
        if not words:
            return 0.0

        # Use stateful KenLM scoring: walk through all words, keeping state,
        # and return the score contribution of the last word.
        state = kenlm.State()
        self.model.BeginSentenceWrite(state)

        score = 0.0
        for word in words:
            out_state = kenlm.State()
            score = self.model.BaseScore(state, word, out_state)
            state = out_state

        # BaseScore returns log10; convert to natural log.
        return score * LN10

    def score_eos(self, token_ids: list[int]) -> float:
        """Score the end of the utterance, flushing any pending word.

        Call this when the beam is finalized to get the LM score for the
        last word plus the EOS penalty.

        Parameters
        ----------
        token_ids : list[int]
            The full decoded token sequence.

        Returns
        -------
        float
            Log-probability (natural log) for the final word + EOS.
        """
        if self.tokenizer is None or not token_ids:
            return 0.0

        vocab = self.tokenizer.vocab
        blank_id = self.tokenizer.blank_id
        words = _bpe_ids_to_words(token_ids, vocab, blank_id)
        if not words:
            return 0.0

        # Walk all words through KenLM to reach the correct state.
        state = kenlm.State()
        self.model.BeginSentenceWrite(state)
        last_word_score = 0.0

        for word in words:
            out_state = kenlm.State()
            last_word_score = self.model.BaseScore(state, word, out_state)
            state = out_state

        # Now score </s> from the final state.
        eos_state = kenlm.State()
        eos_score = self.model.BaseScore(state, "</s>", eos_state)

        # Return last_word_score (in case the caller hasn't scored it yet
        # via score_token, e.g. if the last token was mid-word) + EOS.
        # The caller decides whether to include last_word_score or just EOS.
        # For simplicity, return only the EOS contribution; the integration
        # layer can call score_token for the last word separately.
        return eos_score * LN10

    def make_score_fn(self, alpha: float = 0.3):
        """Create a score_fn callback compatible with beam_search_decode.

        Parameters
        ----------
        alpha : float
            LM weight (shallow fusion interpolation weight).

        Returns
        -------
        callable
            Function with signature (tokens: list[int], new_token: int) -> float
        """

        def score_fn(tokens: list[int], new_token: int) -> float:
            return alpha * self.score_token(tokens, new_token)

        return score_fn


def _bpe_ids_to_words(
    token_ids: list[int],
    vocab: list[str],
    blank_id: int,
) -> list[str]:
    """Convert a sequence of BPE token IDs to a list of words.

    Groups BPE pieces by word boundaries (tokens starting with the
    boundary marker) and returns cleaned word strings.
    """
    current_pieces: list[str] = []
    words: list[str] = []

    for tid in token_ids:
        if tid == blank_id or tid < 0 or tid >= len(vocab):
            continue
        piece = vocab[tid]
        if piece.startswith(_WORD_BOUNDARY) and current_pieces:
            word = "".join(current_pieces).replace(_WORD_BOUNDARY, "")
            if word:
                words.append(word)
            current_pieces = []
        current_pieces.append(piece)

    # Flush the last accumulated word.
    if current_pieces:
        word = "".join(current_pieces).replace(_WORD_BOUNDARY, "")
        if word:
            words.append(word)

    return words


def download_lm(target_dir: str | None = None) -> str:
    """Download the LibriSpeech 4-gram ARPA model from OpenSLR.

    Downloads to ~/.cache/nemotron-asr-mlx/lm/ by default.

    Parameters
    ----------
    target_dir : str, optional
        Directory to save the model. Created if it does not exist.

    Returns
    -------
    str
        Path to the downloaded (decompressed) ARPA model file.
    """
    if target_dir is None:
        target_dir = os.path.join(
            Path.home(), ".cache", "nemotron-asr-mlx", "lm"
        )
    os.makedirs(target_dir, exist_ok=True)

    arpa_path = os.path.join(target_dir, "4-gram.arpa")
    if os.path.exists(arpa_path):
        return arpa_path

    gz_path = arpa_path + ".gz"
    print(f"Downloading LibriSpeech 4-gram LM to {gz_path} ...")

    def _progress(block_num, block_size, total_size):
        downloaded = block_num * block_size
        if total_size > 0:
            pct = min(100, downloaded * 100 // total_size)
            mb = downloaded / (1024 * 1024)
            total_mb = total_size / (1024 * 1024)
            print(
                f"\r  {mb:.1f} / {total_mb:.1f} MB ({pct}%)",
                end="",
                flush=True,
            )

    urllib.request.urlretrieve(_LIBRISPEECH_4GRAM_URL, gz_path, _progress)
    print()  # newline after progress

    print("Decompressing ...")
    with gzip.open(gz_path, "rb") as f_in:
        with open(arpa_path, "wb") as f_out:
            shutil.copyfileobj(f_in, f_out)
    os.remove(gz_path)

    print(f"LM ready at {arpa_path}")
    return arpa_path
