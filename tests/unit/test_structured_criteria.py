"""Choice options whose descriptions are JSON values rather than strings.

The TypeSafe wire format lets an option description be any JSON value (the Jev Decision Index suite sends
chess moves as {"uci", "san"}, chords as {"names", "pitch_classes"}, palettes as lists of colours). The
gateway must accept them and the semif prompt must show them as one-line JSON, like the request state.
"""

from __future__ import annotations

import pytest

from omj.backends.semif_prompt import render_question
from omj.gateway.schema import parse_request


def _choice(criteria: dict) -> dict:
    return {"type": "choice", "instructions": "Pick one.", "criteria": criteria}


@pytest.mark.parametrize(
    "description",
    [
        {"uci": "c8e8", "san": "Re8"},
        ["#ebf0ff", "#bacce8"],
        3,
        None,
    ],
)
def test_choice_option_description_may_be_any_json_value(description: object) -> None:
    request = parse_request(
        {"model": "m", "state": "s", "questions": {"q": _choice({"a": description, "b": "plain"})}}
    )
    assert request.questions["q"].criteria == {"a": description, "b": "plain"}


def test_structured_descriptions_render_as_one_line_json() -> None:
    body, keys, _labels = render_question(
        {"fen": "8/8"},
        _choice({"c8e8": {"uci": "c8e8", "san": "Re8"}, "pal": ["#ebf0ff", "#bacce8"], "plain": "text"}),
    )
    assert keys == ["c8e8", "pal", "plain"]
    assert 'A. c8e8: {"uci": "c8e8", "san": "Re8"}' in body.splitlines()
    assert 'B. pal: ["#ebf0ff", "#bacce8"]' in body.splitlines()
    assert "C. plain: text" in body.splitlines()


def test_null_description_renders_as_the_option_key() -> None:
    body, _keys, _labels = render_question("s", _choice({"left": None, "right": "go right"}))
    assert "A. left: left" in body.splitlines()
