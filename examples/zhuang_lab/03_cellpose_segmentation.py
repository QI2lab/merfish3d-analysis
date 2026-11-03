"""
Run cellpose, save ROIs, reload ROIs, warp ROIs to global system, save again.

Eventually, this will be re-integrated back into the library - but it is
helpful to see it split out for now.

IMPORTANT: You must optimize the cellpose parameters on your own using the GUI,
then fill in the dictionary at the bottom of the script.

Shepherd 2025/07 - refactor for CellposeSAM
Shepherd 2024/12 - refactor
Shepherd 2024/11 - created script to run cellpose given determined parameters.
"""

from pathlib import Path

import numpy as np
import typer
from cellpose import io, models
from roifile import roiread, roiwrite

from merfish3danalysis.qi2labDataStore import qi2labDataStore
from merfish3danalysis.utils.cellpose_rois import global_rois
from merfish3danalysis.utils.dataio import resolve_datastore_path
from merfish3danalysis.utils.spacing import round_spacing_um

app = typer.Typer(pretty_exceptions_enable=False)


def run_cellpose(root_path: Path, cellpose_parameters: dict) -> None:
    """Run cellpose and save ROIs.

    Parameters
    ----------
    root_path: Path
        path to experiment or qi2labdatastore directory
    cellpose_parameters: dict
        dictionary of cellpose parameters
    """
    # initialize datastore
    datastore_path = resolve_datastore_path(root_path)
    datastore = qi2labDataStore(datastore_path)

    # Load the saved downsampled projection and its fused coordinates.
    projection = datastore.load_global_fiducial_projection(return_future=False)
    if projection is None:
        raise FileNotFoundError("Fused fiducial projection Scene not found.")
    fiducial_max_projection, affine_zyx_um, origin_zyx_um, spacing_zyx_um = projection

    # initialize cellpose model and options
    model = models.CellposeModel(gpu=True)
    normalize = {
        "normalize": True,
        "percentile": cellpose_parameters["normalization"],
    }

    # run cellpose on fiducial max projection
    masks, _, _ = model.eval(
        fiducial_max_projection,
        do_3D=False,
        diameter=cellpose_parameters["diameter"],
        flow_threshold=cellpose_parameters["flow_threshold"],
        cellprob_threshold=-cellpose_parameters["cellprob_threshold"],
        niter=200,
        normalize=normalize,
    )

    # save masks
    downsampling_zyx = round_spacing_um(spacing_zyx_um) / round_spacing_um(
        datastore.voxel_size_zyx_um
    )
    datastore.save_global_cellpose_segmentation_image(
        masks, downsampling=downsampling_zyx.tolist()
    )

    # save pixel spaced ROIs
    cellpose_roi_path = datastore.cellpose_pixel_rois_path()
    cellpose_roi_path.parent.mkdir(parents=True, exist_ok=True)
    imagej_roi_path = cellpose_roi_path.with_name("pixel")
    if np.any(masks):
        io.save_rois(masks, str(imagej_roi_path))
    else:
        roiwrite(cellpose_roi_path, [], mode="w")

    # load pixel spaced ROIs
    pixel_spacing_rois = roiread(cellpose_roi_path)

    global_spacing_rois = list(
        global_rois(
            pixel_spacing_rois,
            spacing_zyx_um,
            origin_zyx_um,
            affine_zyx_um,
            min_cell_area_um2=0,
            workers=1,
        )
    )

    # write global coordinate ROIs
    global_roi_path = datastore.cellpose_global_rois_path()
    roiwrite(global_roi_path, global_spacing_rois, mode="w")

    # update datastore state
    datastore_state = datastore.datastore_state.copy()
    datastore_state.update({"SegmentedCells": True})
    datastore.datastore_state = datastore_state


@app.command()
def main(root_path: Path) -> None:
    """Segment the downsampled fiducial projection with Zhuang parameters."""
    root_path = root_path.expanduser().resolve()
    # Zhuang parameters; tune on the downsampled fiducial projection.
    # The original convention passes the negative of cellprob_threshold.
    cellpose_parameters = {
        "normalization": [0.5, 99.5],
        "flow_threshold": 0.4,
        "cellprob_threshold": 1.0,
        "diameter": 15,
    }
    run_cellpose(root_path, cellpose_parameters)


if __name__ == "__main__":
    app()
