from wf_merfish.postprocess.PixelDecoder import PixelDecoder
from pathlib import Path
from tqdm import tqdm
import gc
import cupy as cp
import numpy as np

data_path = Path('/mnt/data/qi2lab/20240317_OB_MERFISH_7')

dataset_path = data_path / Path("processed_v2")

decode_factory = PixelDecoder(dataset_path=dataset_path,
                              exp_type='3D',
                              verbose=1)

decode_factory.decode_all_tiles(minimum_pixels=27,fdr_target=.05)