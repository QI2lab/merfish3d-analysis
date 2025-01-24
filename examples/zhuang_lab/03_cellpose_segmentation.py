"""
Run cellpose, save ROIs, reload ROIs, warp ROIs to global system, save again.

Eventually, this will be re-integrated back into the library - but it is 
helpful to see it split out for now.

IMPORTANT: You must optimize the cellpose parameters on your own using the GUI,
then fill in the dictionary at the bottom of the script.

Shepherd 2024/12 - refactor
Shepherd 2024/11 - created script to run cellpose given determined parameters.
"""

from merfish3danalysis.qi2labDataStore import qi2labDataStore
from pathlib import Path
import numpy as np
from cellpose import models, io
from roifile import roiread, roiwrite, ImagejRoi

def warp_point(
    pixel_space_point: np.ndarray,
    spacing: np.ndarray,
    origin: np.ndarray,
    affine: np.ndarray
) -> np.ndarray:
    """Warp point from pixel space to global space using known transforms.
    
    Parameters
    ----------
    pixel_space_point : np.ndarray
        point in the image coordinate system, zyx order
    spacing: np.ndarray
        pixel size in microns, zyx order
    origin: np.ndarray
        world coordinate origin (um), zyx order
    affine: np.ndarray
        4x4 affine matrix (um), zyx order
        
    Returns
    -------
    registered_space_point: np.ndarray
        point in the world coordinate system (um), zyx order
    
    """

    physical_space_point = pixel_space_point * spacing + origin
    registered_space_point = (np.array(affine) @ np.array(list(physical_space_point) + [1]))[:-1]
    
    return registered_space_point

def run_cellpose(root_path,
                 cellpose_parameters: dict):
    """Run cellpose and save ROIs

    Parameters
    ----------
    root_path: Path
        path to experiment
    cellpose_parameters: dict
        dictionary of cellpose parameters
    """
    
    # initialize datastore
    datastore_path = root_path / Path(r"qi2labdatastore")
    datastore = qi2labDataStore(datastore_path)
    
    # load downsampled, fused polyDT image and coordinates 
    polyDT_fused, affine_zyx_um, origin_zyx_um, spacing_zyx_um = datastore.load_global_fidicual_image(return_future=False)
    
    # create max projection
    polyDT_max_projection = np.max(np.squeeze(polyDT_fused),axis=0)
    del polyDT_fused
    
    # initialize cellpose model and options
    model = models.Cellpose(gpu=True, model_type="cyto3")
    normalize = {
        "normalize": True,
        "percentile": cellpose_parameters['normalization'],
    }

    # run cellpose on polyDT max projection
    masks, _, _, _ = model.eval(
        polyDT_max_projection,
        diameter=cellpose_parameters['diameter'],
        channels=[0,0],
        flow_threshold=cellpose_parameters['flow_threshold'],
        cellprob_threshold=-cellpose_parameters['cellprob_threshold'],
        niter=0,
        normalize=normalize)
    
    # save masks
    datastore.save_global_cellpose_segmentation_image(
        masks,
        downsampling=[1,3.5,3.5]
    )
    
    # save pixel spaced ROIs
    imagej_roi_path_dir = datastore_path / Path("segmentation") / Path("cellpose") / Path("imagej_rois")
    if not(imagej_roi_path_dir.exists()):
        imagej_roi_path_dir.mkdir()
    imagej_roi_path = imagej_roi_path_dir / Path("pixel_spacing")
    io.save_rois(masks, str(imagej_roi_path))
    
    # load pixel spaced ROIs
    cellpose_roi_path = imagej_roi_path_dir / Path("pixel_spacing_rois.zip")
    pixel_spacing_rois = roiread(cellpose_roi_path)
    
    # warp ROIs into global coordinates
    # the ROIs are in (x,y) format. So we have to (1) fake a z dimension,
    # (2) flip xy to yx, (3) warp, (4) remove the z, (5) flip back to (x,y)
    # When we load to check if RNA are in an ROI, need to remember to flip
    # back to (y,x).
    global_spacing_rois = []
    for cell_idx, pixel_spaced_roi in enumerate(pixel_spacing_rois):
        pixel_coordinates = pixel_spaced_roi.coordinates().astype(np.float32)
        padding = np.full((pixel_coordinates.shape[0], 1), 10)
        padded_pixel_coordinates = np.hstack((padding, pixel_coordinates[:, ::-1]))
        global_coordinates_padded = np.zeros_like(padded_pixel_coordinates,dtype=np.float32)
        for pt_idx, pts in enumerate(padded_pixel_coordinates):
            global_coordinates_padded[pt_idx,:] = warp_point(
                pts.copy().astype(np.float32),
                spacing_zyx_um,
                origin_zyx_um,
                affine_zyx_um
            )
        global_coordinates = global_coordinates_padded[:, 1:]
        roi = ImagejRoi.frompoints(np.round(global_coordinates[:,::-1],2).astype(np.float32))
        roi.name = "cell_"+str(cell_idx).zfill(7)
        global_spacing_rois.append(roi)
        del roi
    
    # write global coordinate ROIs
    global_roi_path = imagej_roi_path_dir / Path("global_coords_rois.zip")
    pixel_spacing_rois = roiwrite(global_roi_path,global_spacing_rois)
        
if __name__ == "__main__":
    root_path = Path(r"/mnt/data/zhuang/")
    cellpose_parameters = {
        'normalization' : [1.0,99.0],
        'flow_threshold' : 0.4,
        'cellprob_threshold' : 0.0,
        'diameter': 37
    }
    run_cellpose(root_path, cellpose_parameters)