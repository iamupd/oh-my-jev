"""Option label table and single-token validation for the semif backend.
# REQ-025
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")

from omj.backends.semif import SemifBackend, single_token_labels  # noqa: E402
from omj.backends.semif_prompt import LABELS  # noqa: E402
from omj.config import BackendSection  # noqa: E402
from omj.errors import ErrorCode, OmjError  # noqa: E402


class FakeTokenizer:
    """Maps every known label to one id, except those listed as multi-token."""

    def __init__(self, multi: set[str] | None = None, single: set[str] | None = None) -> None:
        self._multi = multi or set()
        self._single = single
        self.pad_token = "<pad>"
        self.eos_token = "<eos>"
        self.padding_side = "right"
        self.chat_template = None

    def encode(self, text: str, add_special_tokens: bool = True) -> list[int]:
        assert add_special_tokens is False, "label probing must not add special tokens"
        assert text.startswith(" "), "labels must be probed with a leading space"
        label = text[1:]
        if self._single is not None and label not in self._single:
            return [7, 8]
        if label in self._multi:
            return [7, 8]
        return [100 + LABELS.index(label)]


class FakeModel:
    def parameters(self):
        return iter(())


def _section(**kwargs: object) -> BackendSection:
    defaults: dict = {"name": "semif", "model": "tiny/model", "revision": ""}
    defaults.update(kwargs)
    return BackendSection(**defaults)


def test_labels_are_255_unique_and_ordered() -> None:
    # REQ-025
    assert len(LABELS) == 255
    assert len(set(LABELS)) == 255
    assert LABELS[:3] == ["A", "B", "C"]
    assert LABELS[25] == "Z"
    assert LABELS[26] == "AA"
    assert LABELS[51] == "AZ"
    assert LABELS[52] == "BA"
    assert LABELS[-1] == "IU"
    assert all(label.isupper() and label.isalpha() for label in LABELS)
    assert all(1 <= len(label) <= 2 for label in LABELS)


def test_single_token_labels_drops_multi_token_entries() -> None:
    # REQ-025
    labels, ids = single_token_labels(FakeTokenizer(multi={"AB"}))
    assert "AB" not in labels
    assert "AA" in labels
    assert len(labels) == 254
    assert len(ids) == 254
    assert labels == [label for label in LABELS if label != "AB"]
    assert ids[0] == 100


def test_load_keeps_only_single_token_labels_and_caps_max_options() -> None:
    # REQ-025
    backend = SemifBackend(model=FakeModel(), tokenizer=FakeTokenizer(multi={"AB", "AC"}))
    backend.load(_section(revision="15852e8c16360a2fea060d615a32b45270f8a8fc"))
    assert backend.capabilities.max_options == 253
    assert backend.capabilities.device == "cpu"
    assert backend.model_id == "tiny/model@15852e8c"
    assert backend.health().ok is True


def test_load_defaults_revision_to_main() -> None:
    # REQ-025
    backend = SemifBackend(model=FakeModel(), tokenizer=FakeTokenizer())
    backend.load(_section())
    assert backend.model_id == "tiny/model@main"
    assert backend.capabilities.max_options == 255
    assert backend.capabilities.max_state_tokens == 4096


def test_load_rejects_tokenizer_with_fewer_than_two_single_token_labels() -> None:
    # REQ-025
    backend = SemifBackend(model=FakeModel(), tokenizer=FakeTokenizer(single={"A"}))
    with pytest.raises(OmjError) as exc_info:
        backend.load(_section())
    assert exc_info.value.code == ErrorCode.E_BACKEND
    assert "single token" in exc_info.value.message


def test_health_is_not_ok_before_load() -> None:
    # REQ-025
    backend = SemifBackend()
    assert backend.health().ok is False
