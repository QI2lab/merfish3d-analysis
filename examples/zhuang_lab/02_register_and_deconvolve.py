"""Generate deconvolved data and create "fake" local tile registrations.

In this example, we  bypass the standard "DataRegistration" API because
the Zhuang MOP data is already registered and warped.

For fiducial data, only round 1 is deconvolved. A rigid xyz transform
consisting of all zeros is added to all tiles & rounds for the fiducial data.

For readout data, all tiles and bits are deconvolved plus u-fish predicted.

Shepherd 2024/08 - rework script to utilized qi2labdatastore object.
"""

from pathlib import Path

from merfish3danalysis.qi2labDataStore import qi2labDataStore


def local_register_data(root_path: Path) -> None:
    """Register each tile across rounds in local coordinates.

    Parameters
    ----------
    root_path: Path
        path to experiment
    """
    from merfish3danalysis.DataRegistration import DataRegistration

    # initialize datastore
    datastore_path = root_path / Path(r"qi2labdatastore")
    datastore = qi2labDataStore(datastore_path)

    # initialize registration class
    registration_factory = DataRegistration(
        datastore=datastore,
        bkd_subtract_fiducial=False,
        decon_readout=True,
        perform_optical_flow=False,
        overwrite_registered=True,
        save_all_fiducial_registered=False,
        crop_yx_decon=2048,
    )

    # run local registration across rounds
    registration_factory.register_all_tiles()

    # update datastore state
    datastore_state = datastore.datastore_state
    datastore_state.update({"LocalRegistered": True})
    datastore.datastore_state = datastore_state


def global_register_data(
    root_path: Path, create_max_proj_tiff: bool | None = True
) -> None:
    """Register all tiles in first round in global coordinates.

    Parameters
    ----------
    root_path: Path
        path to experiment

    create_max_proj_tiff: Optional[bool]
        create max projection tiff in the segmentation/cellpose directory.
        Default = True
    """

    from merfish3danalysis.cli.qi2lab_microscopes.global_register import (
        global_register_data as run_global_register_data,
    )

    run_global_register_data(
        root_path=root_path,
        fused_chunk_size=512,
        create_max_proj_tiff=bool(create_max_proj_tiff),
        use_gpu_fusion=True,
        ngff_version="0.5",
    )


if __name__ == "__main__":
    root_path = Path(r"/media/dps/data/zhuang")
    # local_register_data(root_path)
    global_register_data(root_path, create_max_proj_tiff=True)
