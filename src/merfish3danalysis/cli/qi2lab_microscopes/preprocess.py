"""
Perform registration on qi2labdatastore. By default creates a max
projection downsampled fiducial OME-TIFF for cellpose parameter optimization.

Shepherd 2025/10 - change to CLI.
Shepherd 2025/07 - rework for multiple GPU support.
Shepherd 2024/11 - rework script to accept parameters.
Shepherd 2024/08 - rework script to utilized qi2labdatastore object.
"""

from pathlib import Path

import typer

from merfish3danalysis.qi2labDataStore import qi2labDataStore

app = typer.Typer()
app.pretty_exceptions_enable = False


@app.command()
def local_register_data(
    root_path: Path,
    num_gpus: int = 1,
    decon: bool = False,
    opticalflow: bool = True,
    decon_allfiducial: bool = True,
    bkdsubtract_all_fiducial: bool = False,
    save_all_fiducial: bool = False,
    overwrite: bool = True,
    crop_yx_decon: int = 1024,
    zstride_level: int = 0,
    verbose: int = 1,
) -> None:
    """Preprocess and register each tile across rounds in local coordinates.

    Parameters
    ----------
    root_path: Path
        path to experiment
    num_gpus: int, Default = 1
        number of gpus available.
    decon: bool, Default = False
        perform readout deconvolution. If False, corrected data are re-saved for
        compatibility instead of deconvolved readouts.
    opticalflow: bool, Default = True
        perform optical flow based registration.
    decon_allfiducial: bool, Default = True
        perform deconvolution prior to registration for fiducials beyond the first round.
    bkdsubtract_all_fiducial: bool, Default = False
        perform background subtraction prior to registration.
    save_all_fiducial: bool, Default = False
        save all registered fiducial images.
    overwrite: bool, Default = True
        overwrite existing registered data.
    crop_yx_decon: int, default = 1024
        size of tile for GPU deconvolution.
    zstride_level: int, default = 0
        look for a skip z dataset.
    verbose : int, default = 1
        progress verbosity. Set to 0 to suppress routine progress prints.

    """
    from merfish3danalysis.DataRegistration import DataRegistration

    # initialize datastore
    if zstride_level == 0:
        datastore_path = root_path / Path(r"qi2labdatastore")
    else:
        datastore_path = root_path / Path(f"qi2labdatastore_zstride0{zstride_level}")
    datastore = qi2labDataStore(datastore_path)
    print(f"Using datastore at {datastore_path}")

    # initialize registration class
    registration_factory = DataRegistration(
        datastore=datastore,
        decon_fiducial=decon_allfiducial,
        decon_readout=decon,
        bkd_subtract_fiducial=bkdsubtract_all_fiducial,
        perform_optical_flow=opticalflow,
        overwrite_registered=overwrite,
        save_all_fiducial_registered=save_all_fiducial,
        num_gpus=num_gpus,
        crop_yx_decon=crop_yx_decon,
        verbose=verbose,
    )

    # run local registration across rounds
    registration_factory.register_all_tiles()

    # update datastore state
    datastore_state = datastore.datastore_state
    datastore_state.update({"LocalRegistered": True})
    datastore.datastore_state = datastore_state


def main() -> None:
    app()


if __name__ == "__main__":
    main()
