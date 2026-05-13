"""CLI entry point for the qi2lab datastore viewer."""

from __future__ import annotations

from pathlib import Path  # noqa: TC003

import typer

from merfish3danalysis.viewer import Qi2labViewer

app = typer.Typer()
app.pretty_exceptions_enable = False
VIEW_PATH_ARGUMENT = typer.Argument(
    None,
    help="Experiment root or qi2labdatastore path. Opens a picker if omitted.",
)


@app.command()
def view(path: Path | None = VIEW_PATH_ARGUMENT) -> None:
    """Open a view-only ndv GUI for a qi2lab datastore."""

    Qi2labViewer(path).run()


def main() -> None:
    app()


if __name__ == "__main__":
    main()
