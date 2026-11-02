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
def image_datastore(tmp_path, request):
    """A synthetic two-round, two-wavelength acquisition using public APIs."""
    import numpy as np
    import pandas as pd

    from merfish3danalysis.qi2labDataStore import qi2labDataStore

    datastore = qi2labDataStore(tmp_path / "qi2labdatastore")
    six_bits = getattr(request, "param", None) == "six_bits"
    six_tiles = getattr(request, "param", None) == "six_tiles"
    datastore.channels_in_data = ["fiducial", "readout1", "readout2"]
    datastore.voxel_size_zyx_um = [0.4, 0.2, 0.4]
    datastore.num_tiles = 6 if six_tiles else 2
    datastore.experiment_order = np.array(
        [[1, 1, 2], [2, 3, 4], [3, 5, 6]] if six_bits else [[1, 1, 3], [2, 2, 4]]
    )
    datastore.tile_overlap = 0.1
    datastore.binning = 1
    datastore.e_per_ADU = 1.0
    datastore.na = 1.4
    datastore.ri = 1.5
    datastore.microscope_type = "synthetic"
    datastore.camera_model = "synthetic"
    datastore.codebook = pd.DataFrame(
        {
            "gene_id": ["gene"],
            **{f"bit{bit:02}": [1] for bit in range(1, 7 if six_bits else 5)},
        }
    )
    for tile in range(datastore.num_tiles):
        datastore.initialize_tile(tile)
        for bit in range(datastore.num_bits):
            wavelengths = (0.63, 0.67) if bit < 2 else (0.53, 0.58)
            datastore.save_local_wavelengths_um(wavelengths, tile=tile, bit=bit)
    datastore.datastore_state = {"Calibrations": True}
    return datastore


@pytest.fixture
def pending_scene(image_datastore):
    """Use the acquisition Scene initialized by the datastore API."""
    return image_datastore.datastore_path
