import random

from loom_bench.tokenize import SimpleTokenizer, Tokenizer


def test_simple_tokenizer_round_trips_and_counts():
    tok = SimpleTokenizer()
    text = "What is 37 + 48?\n  Answer:  85."
    assert "".join(tok.pieces(text)) == text
    assert tok.count("hello world") == 2
    assert tok.count("123") == 3  # digits split individually


def test_random_text_exact_length_and_deterministic():
    tok = SimpleTokenizer()
    for n in (0, 1, 7, 1000):
        assert tok.count(tok.random_text(n, random.Random(1))) == n
    assert tok.random_text(50, random.Random(3)) == tok.random_text(50, random.Random(3))


def test_truncate():
    tok = SimpleTokenizer()
    assert tok.truncate("a b c d", 2) == "a b"
    assert tok.truncate("a b", 10) == "a b"


def test_protocol():
    assert isinstance(SimpleTokenizer(), Tokenizer)
