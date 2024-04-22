from wf_merfish.postprocess.DataRegistration import DataRegistration
from pathlib import Path
import gc

readout = False
data_to_use = 'both'

data_dir_path = Path('/mnt/opm3/20240416_BiFISH_cleared_singlecolor/processed_v2/')
polyDT_dir_path = data_dir_path / Path('polyDT')
tile_ids = [entry.name for entry in polyDT_dir_path.iterdir() if entry.is_dir()]
tile_ids = ['tile0000']

for tile_idx, tile_id in enumerate(tile_ids):
    data_register_factory = DataRegistration(dataset_path=data_dir_path,
                                                overwrite_registered=False,
                                                perform_optical_flow=True,
                                                tile_idx=tile_idx)
    
    data_register_factory.load_registered_data(readouts=readout, data_to_read=data_to_use)
    data_register_factory._create_figure(readouts=readout, data_to_display=data_to_use)
    
    del data_register_factory
    gc.collect()