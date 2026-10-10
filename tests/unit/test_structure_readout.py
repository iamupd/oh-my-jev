"""Noncausal readout structure: file round trip, label remap, and semif loading from the model directory."""

from __future__ import annotations

from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("safetensors")

from omj import structure  # noqa: E402
from omj.backends.semif import SemifBackend  # noqa: E402
from omj.errors import ErrorCode, OmjError  # noqa: E402


def _save(directory: Path, table: list[int], kind: str = "noncausal_readout") -> torch.Tensor:
    head = structure.ReadoutHead(4, len(table))
    torch.manual_seed(0)
    with torch.no_grad():
        head.proj.weight.normal_()
    structure.save_readout(directory, head, table, structure=kind)
    return head.proj.weight.detach().clone()


def _backend() -> SemifBackend:
    backend = SemifBackend.__new__(SemifBackend)
    backend._model = torch.nn.Linear(2, 2)
    backend._readout = None
    backend._readout_rows = {}
    backend._model_dir = None
    return backend


def test_readout_round_trip(tmp_path: Path) -> None:
    weight = _save(tmp_path, [11, 22, 33])
    head, table, kind = structure.load_readout(tmp_path)
    assert kind == "noncausal_readout" and table == [11, 22, 33]
    assert torch.equal(head.proj.weight, weight)


def test_remap_label_ids_maps_tokens_to_rows_and_rejects_unknown() -> None:
    ids = torch.tensor([[22, 11], [33, 0]])
    mask = torch.tensor([[True, True], [True, False]])
    assert structure.remap_label_ids(ids, mask, [11, 22, 33]).tolist() == [[1, 0], [2, 0]]
    with pytest.raises(ValueError):
        structure.remap_label_ids(torch.tensor([[99]]), torch.tensor([[True]]), [11, 22, 33])


def test_structure_is_read_from_model_directory(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _save(tmp_path, [7, 8])
    calls: list[object] = []
    monkeypatch.setattr(structure, "enable_noncausal", lambda model: calls.append(model))
    backend = _backend()
    backend._model_dir = tmp_path
    backend._load_structure([])
    assert backend._readout is not None
    assert backend._readout_rows == {7: 0, 8: 1}
    assert calls == [backend._model]


def test_adapter_directory_takes_precedence_over_model_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / "adapter").mkdir()
    (tmp_path / "model").mkdir()
    _save(tmp_path / "adapter", [5, 6])
    _save(tmp_path / "model", [1, 2])
    monkeypatch.setattr(structure, "enable_noncausal", lambda model: None)
    backend = _backend()
    backend._model_dir = tmp_path / "model"
    backend._load_structure([tmp_path / "adapter"])
    assert backend._readout_rows == {5: 0, 6: 1}


def test_no_structure_files_keeps_the_causal_path(tmp_path: Path) -> None:
    backend = _backend()
    backend._model_dir = tmp_path
    backend._load_structure([])
    assert backend._readout is None and backend._readout_rows == {}


def test_causal_structure_file_is_ignored(tmp_path: Path) -> None:
    _save(tmp_path, [1, 2], kind="causal")
    backend = _backend()
    backend._model_dir = tmp_path
    backend._load_structure([])
    assert backend._readout is None


def test_corrupt_readout_is_a_backend_error(tmp_path: Path) -> None:
    (tmp_path / structure.CONFIG_FILE).write_text("{not json", encoding="utf-8")
    backend = _backend()
    backend._model_dir = tmp_path
    with pytest.raises(OmjError) as err:
        backend._load_structure([])
    assert err.value.code == ErrorCode.E_BACKEND
