import os
import sys
import stat
from pathlib import Path
import typer

app = typer.Typer()
app.pretty_exceptions_enable = False

@app.command()
def setup_activation():
    """
    Create a conda activation script that appends NVIDIA library paths to LD_LIBRARY_PATH.
    """
    # Get current conda environment prefix
    conda_prefix = os.environ.get('CONDA_PREFIX')
    if not conda_prefix:
        typer.echo("Error: CONDA_PREFIX is not set. Are you inside a conda environment?", err=True)
        raise typer.Exit(code=1)

    # Determine current Python version directory
    py_ver = f"python{sys.version_info.major}.{sys.version_info.minor}"
    # Paths for activation scripts
    activate_dir = Path(conda_prefix) / 'etc' / 'conda' / 'activate.d'
    activate_dir.mkdir(parents=True, exist_ok=True)

    # Shell script to add NVIDIA lib dirs
    script_path = activate_dir / 'nvidia_ld_library_path.sh'
    script_content = f"""#!/bin/sh
# Automatically added: prepend NVIDIA lib directories to LD_LIBRARY_PATH
for d in $(find $CONDA_PREFIX/lib/{py_ver}/site-packages/nvidia -type d -name lib); do
    export LD_LIBRARY_PATH="$d:$LD_LIBRARY_PATH"
done
"""
    # Write and make executable
    script_path.write_text(script_content)
    script_path.chmod(script_path.stat().st_mode | stat.S_IEXEC)

    typer.echo(f"Activation script created at {script_path}")

def main():
    """
    Main function to run the Typer app.
    """
    app()

if __name__ == "__main__":
    main()