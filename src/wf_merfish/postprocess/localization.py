from cmap import Colormap
from pathlib import Path
from numpy.typing import NDArray
from typing import List, Dict, Generator
import numpy as np
import zarr
from superqt.utils import thread_worker
import gc
from spots3d import spots3d

@thread_worker
def visualize_tile(dataset_path: Path):
    """
    visualize registered polyDT for one tile
    """
    
    round_ids = [entry.name.split('.')[0]  for entry in dataset_path.iterdir() if entry.is_dir()]
    
    data_registered = []
    try:
        for round_id in round_ids:
            current_round_path = dataset_path / Path(round_id + ".zarr")
            current_round = zarr.open(current_round_path,mode='r')
            data_registered.append(np.asarray(current_round["registered_data"], dtype=np.uint16))
                    
        data_registered = np.stack(data_registered,axis=0)
        voxel_zyx_um = np.asarray(current_round.attrs['voxel_zyx_um'], dtype=np.float32)
        del data_registered, current_round
        gc.collect()
    except Exception:
        yield None
        
    create_figure(data_registered,round_ids,voxel_zyx_um)
    
    yield True

@thread_worker
def localize_tile(dataset_path: Path):
    """
    interactive localize all bits for one tile
    """
    pass

@thread_worker
def batch_localize(dataset_path: Path) -> Generator[Dict[str, int], None, None]:
    """
    automated localize all bits for all tiles
    """
    pass

def create_figure(data_registered: NDArray,
                  round_ids: List[str],
                  voxel_zyx_um: NDArray):
    """
    Generate napari figure
    """
    import napari
    
    viewer = napari.Viewer()
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
                         contrast_limits=[100,np.percentile(data_registered[r_idx][middle_slice,:].ravel(),99.98)])

    napari.run()