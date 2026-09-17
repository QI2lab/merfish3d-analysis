"""Register Zhuang MOP data and generate U-FISH predictions.

The BIL data are already locally registered and warped. This example keeps
deconvolution and deformable registration disabled. Global registration uses
the shared DataRegistration workflow and writes downsampled fiducial fusion
for segmentation.

Shepherd 2024/08 - rework script to utilized qi2labdatastore object.
"""

from pathlib import Path

import typer

from merfish3danalysis.qi2labDataStore import qi2labDataStore
from merfish3danalysis.utils.dataio import resolve_datastore_path

app = typer.Typer(pretty_exceptions_enable=False)


def local_register_data(root_path: Path) -> None:
    """Register each tile across rounds in local coordinates.

    Parameters
    ----------
    root_path: Path
        path to experiment or qi2labdatastore directory
    """
    from merfish3danalysis.DataRegistration import DataRegistration

    # initialize datastore
    datastore_path = resolve_datastore_path(root_path)
    datastore = qi2labDataStore(datastore_path)

    # initialize registration class
    registration_factory = DataRegistration(
        datastore=datastore,
        decon_fiducial=False,
        decon_readout=False,
        perform_deformable_registration=False,
        overwrite_outputs=True,
        crop_yx_decon=2048,
    )

    # run local registration across rounds
    registration_factory.register_all_tiles()

    # update datastore state
    datastore_state = datastore.datastore_state.copy()
    datastore_state.update({"LocalRegistered": True})
    datastore.datastore_state = datastore_state


def global_register_data(
    root_path: Path, create_max_proj_tiff: bool | None = True
) -> None:
    """Register first-round tiles and write downsampled fusion.

    Parameters
    ----------
    root_path: Path
        path to experiment or qi2labdatastore directory

    create_max_proj_tiff: Optional[bool]
        create max projection tiff in the segmentation/cellpose directory.
        Default = True
    """

    from merfish3danalysis.DataRegistration import (
        DataRegistration,
        GlobalRegistrationConfig,
    )

    datastore_path = resolve_datastore_path(root_path)
    datastore = qi2labDataStore(datastore_path, validate=False)
    registration_factory = DataRegistration(
        datastore=datastore,
        perform_deformable_registration=False,
        global_registration=True,
        global_registration_config=GlobalRegistrationConfig(
            registration_binning_zyx=(1, 3, 3),
        ),
    )
    registration_factory.global_register(
        create_max_proj_tiff=bool(create_max_proj_tiff)
    )


@app.command()
def main(root_path: Path, local_only: bool = False, global_only: bool = False) -> None:
    """Run local and global Zhuang registration, or only the selected stage."""
    if local_only and global_only:
        raise typer.BadParameter(
            "--local-only and --global-only are mutually exclusive."
        )
    root_path = root_path.expanduser().resolve()
    if not global_only:
        local_register_data(root_path)
    if not local_only:
        global_register_data(root_path, create_max_proj_tiff=True)


if __name__ == "__main__":
    app()
