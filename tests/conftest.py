from pathlib import Path
from unittest.mock import Mock

import pytest


def pytest_addoption(parser: pytest.Parser) -> None:
    """Register dataset and optional integration-matrix command-line options."""
    parser.addoption(
        "--run-simulation-exhaustive",
        action="store_true",
        default=False,
        help="Run the exhaustive local simulation matrix tests.",
    )
    parser.addoption(
        "--simulation-data-root",
        action="store",
        default=None,
        help="Path to the extracted Zenodo simulation dataset.",
    )
    parser.addoption(
        "--sofima-output-png",
        action="store",
        default=None,
        help="Optional path for writing the SOFIMA deformable recovery PNG.",
    )


def pytest_configure(config: pytest.Config) -> None:
    """Register the unit and integration categories and matrix execution option."""
    config.addinivalue_line("markers", "unit: isolated component behavior")
    config.addinivalue_line(
        "markers", "integration: behavior across connected components"
    )
    config.addinivalue_line(
        "markers", "gpu: requires local GPU execution; excluded from GitHub CI"
    )
    config.addinivalue_line(
        "markers",
        "simulation_exhaustive: exhaustive local simulation matrix test",
    )


def pytest_collection_modifyitems(
    config: pytest.Config, items: list[pytest.Item]
) -> None:
    """Require one test category and select the optional simulation matrix."""
    for item in items:
        markers = {marker.name for marker in item.iter_markers()}
        categories = markers & {"unit", "integration"}
        if len(categories) != 1 or markers & {"regression", "smoke", "api"}:
            raise pytest.UsageError(
                f"{item.nodeid} must have exactly one test category: "
                "@pytest.mark.unit or @pytest.mark.integration; "
                "regression, smoke and API categories are not allowed."
            )

    if config.getoption("--run-simulation-exhaustive"):
        return

    skip_exhaustive = pytest.mark.skip(reason="requires --run-simulation-exhaustive")
    for item in items:
        if "simulation_exhaustive" in item.keywords:
            item.add_marker(skip_exhaustive)


@pytest.fixture
def mock_datastore(monkeypatch):
    """One-tile datastore with file reads, writes and existence checks mocked."""
    from merfish3danalysis.qi2labDataStore import qi2labDataStore

    monkeypatch.setattr(Path, "exists", Mock(return_value=False))
    monkeypatch.setattr(Path, "is_file", Mock(return_value=False))
    monkeypatch.setattr(Path, "mkdir", Mock())
    monkeypatch.setattr(qi2labDataStore, "_load_from_json", Mock(return_value={}))
    monkeypatch.setattr(qi2labDataStore, "_save_to_json", Mock())
    monkeypatch.setattr(
        qi2labDataStore, "_load_calibrations_attributes", Mock(return_value={})
    )
    datastore = qi2labDataStore(Path("/mock/qi2labdatastore"))
    datastore.num_tiles = 1
    datastore.num_rounds = 1
    datastore.num_bits = 1
    datastore._round_ids = ["round001"]
    datastore._bit_ids = ["bit001"]
    datastore.voxel_size_zyx_um = [0.4, 0.2, 0.4]
    datastore._save_to_json.reset_mock()
    return datastore
