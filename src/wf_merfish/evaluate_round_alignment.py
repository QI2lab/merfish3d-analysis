from wf_merfish.postprocess.DataRegistration import DataRegistration
from pathlib import Path
import gc

data_dir_path = Path('/mnt/opm3/20240124_OB_Full_MERFISH_UA_3_allrds/processed_v2')
polyDT_dir_path = data_dir_path / Path('polyDT')
tile_ids = [entry.name for entry in polyDT_dir_path.iterdir() if entry.is_dir()]

for tile_idx, tile_id in enumerate(tile_ids):
    data_register_factory = DataRegistration(dataset_path=data_dir_path,
                                                overwrite_registered=False,
                                                perform_optical_flow=True,
                                                tile_idx=tile_idx)
    
    data_register_factory.load_registered_data(readouts=True)
    data_register_factory._create_figure(readouts=True)
    
    del data_register_factory
    gc.collect()