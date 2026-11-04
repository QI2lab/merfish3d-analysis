# Contributing

This is an early project and we welcome all contributions! The easiest way to get started is to open an [issue](https://github.com/QI2lab/merfish3d-analysis/issues). We are especially interested in open-source iterative multiplexing datasets that we can setup as examples for this library.

## Testing

Use unit tests for individual behaviors and integration tests for complete
workflows. Mark each test with exactly one of `@pytest.mark.unit` or
`@pytest.mark.integration`; smoke, regression, and API categories are not used.
Reuse fixtures and the library and upstream APIs
when creating data and datastores. Use mocks only where needed to isolate a unit.
Integration tests should exercise real producers and consumers.

Check results against independently specified expectations, including physical
units, axis order, and appropriate tolerances for scientific calculations.
Use `tests/test_*.py` for test files.

```bash
uv run pytest tests/test_*.py -m unit
uv run pytest tests/test_*.py -m integration
uv run pytest tests/test_*.py
```

Mark tests requiring CUDA with `@pytest.mark.gpu`. GitHub CI excludes these tests;
run them locally with a CUDA device:

```bash
uv run pytest tests/test_*.py -m gpu
```

The [simulation example](examples/statphysbio_synthetic.md) describes how to
run the published-data integration tests. These require `--simulation-data-root`;
pass `--run-simulation-exhaustive` to include the full simulation matrix.
