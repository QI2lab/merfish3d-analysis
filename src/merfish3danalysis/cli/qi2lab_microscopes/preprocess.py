"""
Perform registration on qi2labdatastore. By default creates a max.

projection downsampled fiducial OME-TIFF for cellpose parameter optimization.

Shepherd 2025/10 - change to CLI.
Shepherd 2025/07 - rework for multiple GPU support.
Shepherd 2024/11 - rework script to accept parameters.
Shepherd 2024/08 - rework script to use qi2labdatastore object.
"""

from pathlib import Path

import typer

from merfish3danalysis.cli.qi2lab_microscopes._common import qi2lab_datastore_path
from merfish3danalysis.qi2labDataStore import qi2labDataStore

app = typer.Typer()
app.pretty_exceptions_enable = False


@app.command()
def local_register_data(
    root_path: Path,
    num_gpus: int = 1,
    decon: bool = True,
    overwrite: bool = True,
    global_registration_only: bool = False,
    global_fusion_only: bool = False,
    fiducial_registration_only: bool = False,
    verbose: int = 1,
) -> None:
    """Preprocess and register each tile across rounds in local coordinates.

    Parameters
    ----------
    root_path : Path
        Experiment root directory.
    num_gpus : int, default=1
        Number of GPUs available.
    decon : bool, default=True
        Perform readout deconvolution. Use ``--no-decon`` to process corrected
        readout images without deconvolution.
    overwrite : bool, default=True
        Overwrite existing registered data.
    global_registration_only : bool, default=False
        Skip local preprocessing and rerun only global tile registration and
        fused fiducial OME-Zarr creation on an existing datastore.
    global_fusion_only : bool, default=False
        Skip local preprocessing and fuse using stored global tile transforms.
        If transforms are absent, globally register round-1 fiducials first.
    fiducial_registration_only : bool, default=False
        Recompute local affine transforms, SOFIMA fields, global transforms,
        and global fiducial fusion without rerunning readout deconvolution or
        U-FISH.
    verbose : int, default=1
        Progress verbosity. Set to 0 to suppress routine progress prints.

    """
    from merfish3danalysis.DataRegistration import DataRegistration

    selected_modes = sum(
        (
            global_registration_only,
            global_fusion_only,
            fiducial_registration_only,
        )
    )
    if selected_modes > 1:
        raise typer.BadParameter(
            "--global-registration-only, --global-fusion-only, and "
            "--fiducial-registration-only are mutually exclusive."
        )
    # initialize datastore
    datastore_path = qi2lab_datastore_path(root_path)
    datastore = qi2labDataStore(datastore_path)
    print(f"Using datastore at {datastore_path}")

    # initialize registration class
    registration_factory = DataRegistration(
        datastore=datastore,
        decon_fiducial=True,
        decon_readout=decon,
        perform_deformable_registration=True,
        overwrite_outputs=overwrite,
        num_gpus=num_gpus,
        global_registration=True,
        verbose=verbose,
    )

    if global_registration_only:
        registration_factory.global_register(create_max_proj_tiff=True)
        return
    if global_fusion_only:
        registration_factory.fuse_global_registered(create_max_proj_tiff=True)
        return

    # run local registration across rounds
    registration_factory.register_all_tiles(
        process_readouts=not fiducial_registration_only
    )

    # update datastore state
    datastore_state = datastore.datastore_state
    datastore_state.update({"LocalRegistered": True})
    datastore.datastore_state = datastore_state


def main() -> None:
    """
    Run the qi2lab preprocessing command-line application.

    Returns
    -------
    None
        Typer parses the command line and runs the selected preprocessing
        workflow.
    """
    app()


if __name__ == "__main__":
    main()
