"""Decode the Zhuang dataset with its acquisition-specific settings."""

from functools import partial
from pathlib import Path

import typer

from merfish3danalysis.cli.qi2lab_microscopes.pixeldecode import (
    decode_pixels as _decode_pixels,
)

decode_pixels = partial(
    _decode_pixels,
    merfish_bits=22,
    estimate_chromatic_affines=True,
    decode_mode="2d",
)
app = typer.Typer(pretty_exceptions_enable=False)


@app.command()
def main(root_path: Path, reprocess_existing: bool = False) -> None:
    """Decode the Zhuang experiment using its dataset-specific parameters."""
    root_path = root_path.expanduser().resolve()
    decode_pixels(root_path=root_path, reprocess_existing=reprocess_existing)


if __name__ == "__main__":
    app()
