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
    deformable_registration: bool = True,
    decon_allfiducial: bool = True,
    save_all_fiducial: bool = False,
    overwrite: bool = True,
    crop_yx_decon: int = 2048,
    ufish_model: str | None = None,
    global_registration: bool = True,
    global_registration_only: bool = False,
    create_max_proj_tiff: bool = True,
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
    decon: bool, Default = True
        perform readout deconvolution. If False, corrected data are re-saved for
        compatibility instead of deconvolved readouts.
    deformable_registration: bool, Default = True
        perform SOFIMA residual deformable registration.
    decon_allfiducial: bool, Default = True
        perform deconvolution prior to registration for fiducials beyond the first round.
    save_all_fiducial: bool, Default = False
        save all registered fiducial images.
    overwrite: bool, Default = True
        overwrite existing registered data.
    crop_yx_decon: int, default = 2048
        size of tile for GPU deconvolution.
    ufish_model: Optional[str], default = None
        U-FISH model used for feature prediction. If omitted or None, use the
        package default model, simfish. Known aliases include simfish/smfish,
        merfish, seqfish, deepspot, and exseq. A local .onnx/.pth path or
        HuggingFace weights filename may also be used.
    global_registration: bool, default = True
        perform global tile registration and fused fiducial OME-Zarr creation
        after local preprocessing.
    global_registration_only : bool, default=False
        Skip local preprocessing and rerun only global tile registration and
        fused fiducial OME-Zarr creation on an existing datastore.
    create_max_proj_tiff : bool, default=True
        If True, write the fused fiducial max-projection TIFF when global
        registration runs.
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
        perform_deformable_registration=deformable_registration,
        overwrite_registered=overwrite,
        save_all_fiducial_registered=save_all_fiducial,
        num_gpus=num_gpus,
        crop_yx_decon=crop_yx_decon,
        ufish_model=ufish_model,
        global_registration=global_registration,
        verbose=verbose,
    )

    if global_registration_only:
        registration_factory.global_register(create_max_proj_tiff=create_max_proj_tiff)
        return

    # run local registration across rounds
    registration_factory.register_all_tiles()

    # update datastore state
    datastore_state = datastore.datastore_state
    datastore_state.update({"LocalRegistered": True})
    datastore.datastore_state = datastore_state


def main() -> None:
    """
    Main.

    Returns
    -------
    None
        Function result.
    """
    app()


if __name__ == "__main__":
    main()
