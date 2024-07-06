from wf_merfish.postprocess.DataRegistration import DataRegistration
from pathlib import Path
import gc

readout = True # options are 'false' for all rounds polyDT or 'true' for 1st round polyDT + all readout bits
data_to_use = 'predict' # options are 'ufish', 'decon', or 'predict' for ufish, deconvolved, or ufish*deconvolved data

data_path = Path('/mnt/data/qi2lab/20240317_OB_MERFISH_7')
dataset_path = data_path / Path('processed_v2')
polyDT_dir_path = dataset_path / Path('polyDT')
tile_ids = [entry.name for entry in polyDT_dir_path.iterdir() if entry.is_dir()]

for tile_idx, tile_id in enumerate(tile_ids):
    data_register_factory = DataRegistration(dataset_path=dataset_path,
                                                overwrite_registered=False,
                                                perform_optical_flow=True,
                                                tile_idx=tile_idx)
    
    data_register_factory.load_registered_data(readouts=readout, data_to_read=data_to_use)
    data_register_factory.create_figure(readouts=readout, data_to_display=data_to_use)
    
    del data_register_factory
    gc.collect()