import json

import pytest

from routewise.grading import grade
from routewise.tasks import generate


def test_generation_is_deterministic_and_balanced():
    a, b = generate(100, seed=1), generate(100, seed=1)
    assert [t.prompt for t in a] == [t.prompt for t in b]
    assert {t.type for t in a} == {"arithmetic", "word_problem", "extraction", "date", "counting"}
    assert {t.difficulty for t in a} == {"easy", "medium", "hard"}


@pytest.mark.parametrize("t", generate(60, seed=3), ids=lambda t: t.id)
def test_ground_truth_grades_correct(t):
    assert grade(t, t.answer)


def test_grader_tolerates_formatting():
    t = next(t for t in generate(50) if t.type == "arithmetic")
    assert grade(t, f"The answer is **{int(t.answer):,}**.")
    assert not grade(t, str(int(t.answer) + 1))
    ex = next(t for t in generate(50) if t.type == "extraction")
    d = json.loads(ex.answer)
    assert grade(ex, "```json\n" + json.dumps({**d, "amount": f"${d['amount']:,.2f}"}) + "\n```")
    assert not grade(ex, json.dumps({**d, "date": "2020-01-01"}))
    assert not grade(ex, "not json")
