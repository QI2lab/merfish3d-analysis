from cmap import Colormap
from pathlib import Path
from numpy.typing import NDArray
from typing import List, Dict, Generator
import numpy as np
import zarr
from superqt.utils import thread_worker
import gc
from spots3d import spots3d
import napari

@thread_worker
def visualize_tile(dataset_path: Path,
                   tile_id: str):
    """
    visualize registered polyDT for one tile

    Parameters
    ----------
    dataset_path: Path
        root path of qi2lab MERFISH file structure
    tile_id: str
        current tile id
    """

    polyDT_dir_path = dataset_path / Path("polyDT") / Path(tile_id)
    
    round_ids = [entry.name.split('.')[0]  for entry in polyDT_dir_path.iterdir() if entry.is_dir()]
    
    data_registered = []
    try:
        for round_id in round_ids:
            current_round_path = polyDT_dir_path / Path(round_id + ".zarr")
            current_round = zarr.open(current_round_path,mode='r')
            data_registered.append(np.asarray(current_round["registered_data"], dtype=np.uint16))
                    
        data_registered = np.stack(data_registered,axis=0)
        voxel_zyx_um = np.asarray(current_round.attrs['voxel_zyx_um'], dtype=np.float32)
                
        create_figure(data_registered,round_ids,voxel_zyx_um)
        
        del data_registered, 
        gc.collect()
    except Exception:
        yield False

    yield True

@thread_worker
def localize_tile(dataset_path: Path,
                  tile_id: str):
    """
    interactively generate localizations parameters for all bits in one tile

    Parameters
    ----------
    dataset_path: Path
        root path of qi2lab MERFISH file structure
    tile_id: str
        current tile id

    Yields
    ------
    bit iteration as percent of 100
    """

    readout_dir_path = dataset_path / Path("readouts") / Path(tile_id)
    calibration_dir_path = dataset_path / Path("calibrations")
    calibration_zarr = zarr.open(calibration_dir_path,mode='r')
    psfs = calibration_zarr['psfs']
    

    bit_ids = [entry.name.split('.')[0]  for entry in readout_dir_path.iterdir() if entry.is_dir()]

    for bit_id in bit_ids:
        current_bit_dir_path = readout_dir_path / Path(bit_id + ".zarr")
        current_bit_zarr = zarr.open(current_bit_dir_path,mode='r')
        raw_data = np.asarray(current_bit_zarr["raw_data"], dtype=np.uint16)
        em_wvl = float(current_bit_zarr['emission_um'])
        voxel_zyx_um = np.asarray(current_bit_zarr['voxel_zyx_um'])
        if em_wvl == .580:
            psf = psfs[1,:]
        elif em_wvl == .670:
            psf = psfs[2,:]
        else:
            psf = None

        with napari.Viewer() as viewer:
            viewer.add_image(data=raw_data,
                            name=bit_id,
                            scale=voxel_zyx_um)

            dock_widget, plugin_widget = viewer.window.add_plugin_dock_widget("napari-spot-detection", "Spot detection")

            plugin_widget.txt_ri.setText('1.51')
            plugin_widget.txt_lambda_em.setText(str(em_wvl*100))
            plugin_widget.txt_dc.setText(str(voxel_zyx_um[1]))
            plugin_widget.txt_dstage.setText(str(voxel_zyx_um[0]))
            plugin_widget.chk_skewed.setChecked(False)
            path_save_dir = readout_dir_path / Path('localizations') / Path(bit_id)
            path_save_dir.mkdir(parents=True, exist_ok=True) 
            plugin_widget.path_save = path_save_dir / Path ('spots3d_parameters')
            plugin_widget.psf = psf
            plugin_widget.steps_performed['load_psf'] = True
            plugin_widget.but_load_model.click()
            plugin_widget.txt_deconv_iter.setText('50')
            plugin_widget.but_run_deconvolution.click()
            plugin_widget.lab_dog_choice.value = 'deconvolved'
            plugin_widget.but_dog.click()

            napari.run()

        yield (bit_id / len(bit_ids)) * 100
            
@thread_worker
def batch_localize(dataset_path: Path) -> Generator[Dict[str, int], None, None]:
    """
    batch localize all bits for all tiles
    """
    pass

def create_figure(data_registered: NDArray,
                  round_ids: List[str],
                  voxel_zyx_um: NDArray):
    """
    Generate napari figure
    """
    
    with napari.Viewer() as viewer:
        colormaps = [Colormap('cmap:white'),
                    Colormap('cmap:cyan'),
                    Colormap('cmap:yellow'),
                    Colormap('cmap:red'),
                    Colormap('cmap:green'),
                    Colormap('chrisluts:OPF_Fresh'),
                    Colormap('chrisluts:OPF_Orange'),
                    Colormap('chrisluts:OPF_Purple')]
        
        for r_idx, round_id in enumerate(round_ids):
            middle_slice = data_registered[r_idx].shape[0]//2
            viewer.add_image(data=data_registered[r_idx],
                            name=round_id,
                            scale=voxel_zyx_um,
                            blending='additive',
                            colormap=colormaps[r_idx].to_napari(),
                            contrast_limits=[np.percentile(data_registered[r_idx][middle_slice,:].ravel(),10.00),
                                            1.5*np.percentile(data_registered[r_idx][middle_slice,:].ravel(),99.99)])

        napari.run()