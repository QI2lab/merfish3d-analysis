"""Open fused channels in the datastore NDV viewer."""

import typer

from merfish3danalysis.viewer.fused import view_fused_channels as view_fused

app = typer.Typer(pretty_exceptions_enable=False)
app.command()(view_fused)

if __name__ == "__main__":
    app()
