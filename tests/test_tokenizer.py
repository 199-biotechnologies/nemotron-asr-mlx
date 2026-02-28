"""Tests for NemotronTokenizer — decode, blank filtering, from_config."""

from nemotron_asr_mlx.tokenizer import NemotronTokenizer, BLANK_ID


def test_decode_basic():
    vocab = ["\u2581Hello", "\u2581world", "!"]
    tok = NemotronTokenizer(vocab)
    assert tok.decode([0, 1, 2]) == "Hello world!"


def test_decode_blank_filtering():
    vocab = ["\u2581a", "\u2581b", "\u2581c"]
    tok = NemotronTokenizer(vocab)
    result = tok.decode([0, BLANK_ID, 1, BLANK_ID, 2])
    assert result == "a b c"


def test_decode_empty():
    vocab = ["\u2581a"]
    tok = NemotronTokenizer(vocab)
    assert tok.decode([]) == ""


def test_decode_all_blanks():
    vocab = ["\u2581a"]
    tok = NemotronTokenizer(vocab)
    assert tok.decode([BLANK_ID, BLANK_ID]) == ""


def test_decode_out_of_range():
    vocab = ["\u2581a", "\u2581b"]
    tok = NemotronTokenizer(vocab)
    # Out-of-range IDs should be skipped
    assert tok.decode([0, 9999, 1]) == "a b"


def test_from_config_flat():
    config = {"vocab": ["\u2581hi", "\u2581there"]}
    tok = NemotronTokenizer.from_config(config)
    assert tok.decode([0, 1]) == "hi there"


def test_from_config_nested():
    config = {"decoder": {"vocabulary": ["\u2581yes", "\u2581no"]}}
    tok = NemotronTokenizer.from_config(config)
    assert tok.decode([0, 1]) == "yes no"


def test_from_config_custom_blank():
    config = {"vocab": ["\u2581x", "\u2581y"], "blank_id": 99}
    tok = NemotronTokenizer.from_config(config)
    assert tok.blank_id == 99
    assert tok.decode([0, 99, 1]) == "x y"


def test_sentencepiece_underscore_handling():
    vocab = ["\u2581The", "\u2581quick", "\u2581brown", "fox"]
    tok = NemotronTokenizer(vocab)
    # Leading underscore becomes space, then strip removes leading space
    result = tok.decode([0, 1, 2, 3])
    assert result == "The quick brownfox"
