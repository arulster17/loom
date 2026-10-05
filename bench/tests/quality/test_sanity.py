from loom_bench.quality.sanity import (
    SanityLimits,
    check_completions,
    check_output,
    dup_ngram_ratio,
    foreign_script_ratio,
    periodic_tail,
)
from loom_bench.quality.tasks.base import Completion, OutputKind

PROSE = (
    "Expansion joints let a bridge deck lengthen and shorten with temperature, so the "
    "structure does not crack or buckle when a hot afternoon follows a cold night."
)


def test_clean_prose_has_no_flags():
    flags = check_output(PROSE, "stop")
    assert not any(flags.model_dump().values())


def test_empty_and_whitespace():
    assert check_output("", "stop").empty
    assert check_output("  \n\t", "stop").empty


def test_truncation_only_when_unexpected():
    assert check_output(PROSE, "length").truncated
    assert not check_output(PROSE, "length", length_expected=True).truncated
    assert not check_output(PROSE, None).truncated


def test_periodic_tail():
    assert periodic_tail("intro text " + "abc" * 30, 200) == (3, 90)
    assert periodic_tail("no loop here", 200) is None


def test_repetition_loop_detected():
    looped = PROSE + " I will check again." * 20
    assert check_output(looped, "length").repetition
    # Too short to count as a loop.
    assert not check_output("Hmm... ok!!", "stop").repetition


def test_repetition_loop_detected_for_code_too():
    code = "def f():\n    return 1\n" + "    pass\n" * 40
    assert check_output(code, "length", kind=OutputKind.CODE).repetition


def test_duplicate_ngram_ratio():
    ratio, count = dup_ngram_ratio("a b c d a b c d a b c d", 4)
    assert count == 9
    assert ratio == 1 - 4 / 9
    # Repeated phrases with varying separators escape the tail check but not the n-gram one.
    text = " ".join(f"the cat sat on the soft warm mat once again {i}." for i in range(30))
    assert check_output(text, "stop").repetition


def test_ngram_check_skipped_for_code_and_json():
    code = "\n".join(f"assert add({i}, 1) == {i + 1}" for i in range(40))
    assert not check_output(code, "stop", kind=OutputKind.CODE).repetition


def test_language_drift():
    ratio, letters = foreign_script_ratio("Hello мир", ["LATIN"])
    assert letters == 8 and ratio == 3 / 8
    drifted = "The answer is: 这是一个关于桥梁伸缩缝的问题，答案涉及温度变化和材料膨胀。"
    assert check_output(drifted, "stop").language_drift
    assert not check_output("Café crème, naïve résumé façade " * 3, "stop").language_drift
    assert not check_output(drifted, "stop", kind=OutputKind.JSON).language_drift


def test_aggregate_rates_and_limits():
    comps = [Completion(f"i{i}", PROSE, "stop") for i in range(95)]
    comps += [Completion(f"e{i}", "", "stop") for i in range(3)]
    comps += [Completion(f"t{i}", PROSE, "length") for i in range(2)]
    res = check_completions(comps)
    assert res.n == 100
    assert res.counts == {"empty": 3, "truncated": 2, "repetition": 0, "language_drift": 0}
    assert res.rates["empty"] == 0.03
    violations = res.violations(SanityLimits())
    assert len(violations) == 1 and violations[0].startswith("empty rate 3.00%")
    assert res.violations(SanityLimits(max_empty=0.05)) == []
