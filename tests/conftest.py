import pytest


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--run-simulation-exhaustive",
        action="store_true",
        default=False,
        help="Run the exhaustive local simulation matrix tests.",
    )


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line(
        "markers",
        "simulation_exhaustive: exhaustive local simulation matrix test",
    )


def pytest_collection_modifyitems(
    config: pytest.Config, items: list[pytest.Item]
) -> None:
    if config.getoption("--run-simulation-exhaustive"):
        return

    skip_exhaustive = pytest.mark.skip(reason="requires --run-simulation-exhaustive")
    for item in items:
        if "simulation_exhaustive" in item.keywords:
            item.add_marker(skip_exhaustive)
