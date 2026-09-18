# Contributing

This is an early project and we welcome all contributions! The easiest way to get started is to open an [issue](https://github.com/QI2lab/merfish3d-analysis/issues). We are especially interested in open-source iterative multiplexing datasets that we can setup as examples for this library.

## Testing

This package uses unit tests and integration tests only. Do not introduce smoke,
regression, or API test categories, suites, or names.

- Mark unit tests with `@pytest.mark.unit`. Exercise an isolated component's
  behavior using known inputs and independently specified expected results.
  Mock file reads, writes and external inputs. Do not initialize real datastores
  in unit-test fixtures.
- Mark integration tests with `@pytest.mark.integration`. Exercise connected
  components and assert the values passed across their boundaries or the
  resulting outputs. Use real implementations for the interaction being tested;
  mock external inputs. Generate images, metadata, masks and spot tables locally;
  use temporary directories for persistence tests that write and reopen them.
  The explicitly requested published-data matrix is optional and requires
  `--simulation-data-root`; default tests do not access it.

Every test must have exactly one category; pytest rejects unclassified tests,
those with both markers, and smoke/regression/API category markers. Keep tests in `tests/test_*.py` files.
The `simulation_exhaustive` marker controls an optional integration-test matrix,
and is not a separate test category.

Tests must assert the intended scientific and workflow behavior. Successful
imports, process startup, or completion alone do not establish correctness.
Compare pixels, coordinates and metrics against an independent mathematical
expectation. Another run of the implementation is not a ground-truth oracle.
State physical units, axis order and any justified numerical tolerance.
Derive expected values from the documented workflow or a known synthetic input,
not by copying the implementation under test. Bug fixes extend the relevant unit
or integration coverage under these same rules.

Core datastore, registration and decoder correctness is established using
generated ground truth, the published simulations, and the standard qi2lab
commands. Zhuang tests cover its import and example-specific parameters; they
do not define core processing behavior.

For example, fiducial fusion unit tests must assert the required downsampled
output spacing. Integration tests must connect fusion, the saved maximum
projection and its OME calibration, segmentation downsampling metadata, and ROI
coordinates in physical units. Mocking Cellpose inference is appropriate for
this integration test, but mocking fusion or the coordinate conversion would
remove the interaction it must verify.

Run `uv run pytest tests/test_*.py -m unit` or
`uv run pytest tests/test_*.py -m integration` to select a category.
Run `uv run pytest tests/test_*.py` for both. Listing test files avoids traversing
generated simulation artifacts under `tests/data`. Tests that require CUDA or the Zenodo
simulation data retain their existing prerequisites and options; see the
[simulation example](examples/statphysbio_synthetic.md).

The CPU CI job executes both categories against real Zarr v3 OME-Zarr stores,
including pixel round trips, write completion, mask scale metadata, fusion
compression, TIFF calibration, and ROI coordinates. It reads the selected
requirements from `pyproject.toml` and uses CPU PyTorch. Cellpose inference is
mocked in the coordinate integration test; the image and ROI persistence is real.

Mark tests that execute on a GPU with `@pytest.mark.gpu` in addition to their
unit or integration category. Mocked GPU operations do not require this marker.
GPU tests run locally; GitHub CI explicitly excludes them with `-m "not gpu"`.
The marker is an execution requirement, not a third test category.

Run GPU checks locally with the full project environment and a CUDA device:

```bash
uv run pytest tests/test_*.py -m gpu
```

These checks use generated fiducials and readout volumes to verify physical
registration shifts, readout warping, decoding thresholds, barcode centroids,
crop offsets, and intensity sums. External image inputs and model inference are
mocked in generated-data tests. Persistence tests write generated temporary
files. Numerical registration, deconvolution, warping, decoding, and
connected-component extraction run on CUDA. The point-source deconvolution
test also checks photon conservation and forward-model reconstruction error.

The published simulation F1 matrix additionally needs `--simulation-data-root`.
A skipped data-dependent matrix is not evidence that its scientific score passed.

To run the standard qi2lab preprocessing and decoding commands on the published
Nyquist simulation, including persisted transforms and ground-truth F1:

```bash
uv run pytest tests/test_simulation_example_pipeline.py::test_qi2lab_commands_decode_simulation_ground_truth \
  --simulation-data-root /path/to/merfish3d_analysis-simulation
```

This integration test uses real inference and processing on temporary outputs.
It selects all transcripts for normalization because this simulation has no
cell segmentation, and supplies the simulation's decoding thresholds explicitly.
