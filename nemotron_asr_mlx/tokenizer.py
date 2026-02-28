"""Simple BPE tokenizer for Nemotron ASR.

Vocab is 1024 BPE tokens with blank at index 1024.
Decode-only (no encoding needed for inference).
"""

BLANK_ID = 1024


class NemotronTokenizer:
    """Decode token IDs to text using the Nemotron BPE vocabulary."""

    def __init__(self, vocab: list[str], blank_id: int = BLANK_ID):
        self.vocab = vocab
        self.blank_id = blank_id

    def decode(self, token_ids: list[int]) -> str:
        """Convert token IDs to text, filtering blanks and joining pieces."""
        pieces = [
            self.vocab[tid]
            for tid in token_ids
            if tid != self.blank_id and 0 <= tid < len(self.vocab)
        ]
        return "".join(pieces).replace("\u2581", " ").strip()

    @classmethod
    def from_config(cls, config: dict) -> "NemotronTokenizer":
        """Build tokenizer from a config dict containing a 'vocab' key.

        Expected format:
            {"vocab": ["<token0>", "<token1>", ...]}
        or nested under decoder:
            {"decoder": {"vocabulary": ["<token0>", ...]}}
        """
        if "vocab" in config:
            vocab = config["vocab"]
        elif "decoder" in config and "vocabulary" in config["decoder"]:
            vocab = config["decoder"]["vocabulary"]
        else:
            raise ValueError(
                "Cannot find vocabulary in config. "
                "Expected 'vocab' or 'decoder.vocabulary' key."
            )
        blank_id = config.get("blank_id", BLANK_ID)
        return cls(vocab=vocab, blank_id=blank_id)
