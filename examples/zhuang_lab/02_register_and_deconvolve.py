"""Generate deconvolved data and create "fake" local tile registrations.

In this example, we  bypass the standard "DataRegistration" API because
the Zhuang MOP data is already registered and warped.

For fiducial data, only round 1 is deconvolved. A rigid xyz transform
consisting of all zeros is added to all tiles & rounds for the fiducial data.

For readout data, all tiles and bits are deconvolved plus u-fish predicted.

Shepherd 2024/08 - rework script to utilized qi2labdatastore object.
"""

import argparse
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
        decon_fiducial=True,
        decon_readout=True,
        perform_deformable_registration=False,
        overwrite_outputs=True,
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

    from merfish3danalysis.DataRegistration import DataRegistration

    datastore = qi2labDataStore(root_path / "qi2labdatastore")
    registration_factory = DataRegistration(
        datastore=datastore,
        perform_deformable_registration=False,
        global_registration=True,
    )
    registration_factory.global_register(
        create_max_proj_tiff=bool(create_max_proj_tiff)
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("root_path", type=Path)
    stages = parser.add_mutually_exclusive_group()
    stages.add_argument("--local-only", action="store_true")
    stages.add_argument("--global-only", action="store_true")
    args = parser.parse_args()
    root_path = args.root_path.expanduser().resolve()
    if not args.global_only:
        local_register_data(root_path)
    if not args.local_only:
        global_register_data(root_path, create_max_proj_tiff=True)
