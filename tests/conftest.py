import pytest


def pytest_addoption(parser: pytest.Parser) -> None:
    """
    Pytest addoption.

    Parameters
    ----------
    parser : pytest.Parser
        Function argument.

    Returns
    -------
    None
        Function result.
    """
    parser.addoption(
        "--run-simulation-exhaustive",
        action="store_true",
        default=False,
        help="Run the exhaustive local simulation matrix tests.",
    )
    parser.addoption(
        "--sofima-output-png",
        action="store",
        default=None,
        help="Optional path for writing the SOFIMA deformable recovery PNG.",
    )


def pytest_configure(config: pytest.Config) -> None:
    """
    Pytest configure.

    Parameters
    ----------
    config : pytest.Config
        Function argument.

    Returns
    -------
    None
        Function result.
    """
    config.addinivalue_line(
        "markers",
        "simulation_exhaustive: exhaustive local simulation matrix test",
    )


def pytest_collection_modifyitems(
    config: pytest.Config, items: list[pytest.Item]
) -> None:
    """
    Pytest collection modifyitems.

    Parameters
    ----------
    config : pytest.Config
        Function argument.
    items : list[pytest.Item]
        Function argument.

    Returns
    -------
    None
        Function result.
    """
    if config.getoption("--run-simulation-exhaustive"):
        return

    skip_exhaustive = pytest.mark.skip(reason="requires --run-simulation-exhaustive")
    for item in items:
        if "simulation_exhaustive" in item.keywords:
            item.add_marker(skip_exhaustive)
