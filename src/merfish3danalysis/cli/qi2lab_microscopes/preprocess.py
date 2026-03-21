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
    decon: bool = True,
    opticalflow: bool = True,
    decon_allfiducial: bool = False,
    bkdsubtract_all_fiducial: bool = True,
    save_all_fiducial: bool = False,
    overwrite: bool = True,
    crop_yx_decon: int = 1024,
) -> None:
    """Preprocess and register each tile across rounds in local coordinates.

    Parameters
    ----------
    root_path: Path
        path to experiment
    num_gpus: int, Default = 1
        number of gpus available.
    decon: bool, Default = True
        perform deconvolution on 1st round fiducial and FISH readout images.
    opticalflow: bool, Default = True
        perform optical flow based registration.
    decon_allfiducial: bool, Default = False
        perform deconvolution prior to registration.
    bkdsubtract_all_fiducial: bool, Default = True
        perform background subtraction prior to registration.
    save_all_fiducial: bool, Default = False
        save all registered fiducial images.
    overwrite: bool, Default = True
        overwrite existing registered data.
    crop_yx_decon: int, default = 1024
        size of tile for GPU deconvolution.

    """
    from merfish3danalysis.DataRegistration import DataRegistration

    # initialize datastore
    datastore_path = root_path / Path(r"qi2labdatastore")
    datastore = qi2labDataStore(datastore_path)

    # initialize registration class
    registration_factory = DataRegistration(
        datastore=datastore,
        decon_fiducial=decon_allfiducial,
        bkd_subtract_fiducial=bkdsubtract_all_fiducial,
        perform_optical_flow=opticalflow,
        overwrite_registered=overwrite,
        save_all_fiducial_registered=save_all_fiducial,
        num_gpus=num_gpus,
        crop_yx_decon=crop_yx_decon,
    )

    if not (decon):
        registration_factory._decon = False

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
