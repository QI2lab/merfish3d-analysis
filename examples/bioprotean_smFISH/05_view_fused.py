"""
View fused channels using neuroglancer.

Shepherd 2025/03 - created script.
"""

import warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.simplefilter("ignore", category=FutureWarning)
import napari
from pathlib import Path
from cmap import Colormap
from multiview_stitcher import vis_utils
import multiprocessing as mp

mp.set_start_method('spawn', force=True)


def view_fused(root_path: Path):
    """Load and view all individual channels using neuroglancer.
    
    Parameters
    ----------
    root_path: Path
        path to experiment
    """
    
    # generate 17 colormaps
    colormaps = [
        Colormap("cmap:white").to_napari(),
        Colormap("cmap:magenta").to_napari(),
        Colormap("cmap:cyan").to_napari(),
        Colormap("cmap:red").to_napari(),
        Colormap("cmap:yellow").to_napari(),
        Colormap("cmasher:cosmic").to_napari(),
        Colormap("cmasher:dusk").to_napari(),
        Colormap("cmasher:eclipse").to_napari(),
        Colormap("cmasher:emerald").to_napari(),
        Colormap("chrisluts:BOP_Orange").to_napari(),
        Colormap("cmasher:sapphire").to_napari(),
        Colormap("chrisluts:BOP_Blue").to_napari(),
        Colormap("cmap:magenta").to_napari(),
        Colormap("cmap:cyan").to_napari(),
        Colormap("cmap:red").to_napari(),
        Colormap("cmap:yellow").to_napari(),
        Colormap("cmasher:cosmic").to_napari(),
    ]
    
    # find all ome-zarr paths
    ome_path = root_path / Path("fused")
    omezarr_paths = sorted(ome_path.glob("*.ome.zarr"))
    
    # populate napari viewer with all channels
    viewer = napari.Viewer()
    for ch_idx, omezarr_path in enumerate(omezarr_paths):
        # use different contrast limits for polyDT vs FISH channels
        if ch_idx == 0:
            contrast_limits = [0,1000]
        else:
            contrast_limits = [10,500]
        viewer.open(
            str(omezarr_path),
            plugin="napari-ome-zarr",
            blending="additive",
            colormap = colormaps[ch_idx],
            contrast_limits=contrast_limits
        )
    napari.run()
    
if __name__ == "__main__":
    root_path = Path(r"/mnt/data2/bioprotean/20250220_Bartelle_control_smFISH_TqIB")
    view_fused(root_path)