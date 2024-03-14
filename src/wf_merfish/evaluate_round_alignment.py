from wf_merfish.postprocess.DataRegistration import DataRegistration
from pathlib import Path
import gc

readout = True

#data_dir_path = Path('/home/qi2lab/Documents/github/wf-merfish/examples/simulated_images/cylinder/images/jitter-0_shift_amp-0.2_prop_fn-0.1_prop_fp-0.7/processed')
data_dir_path = Path('/mnt/opm3/20240214_MouseBrain_UA_NewRO_RK/processed_v2')
polyDT_dir_path = data_dir_path / Path('polyDT')
tile_ids = [entry.name for entry in polyDT_dir_path.iterdir() if entry.is_dir()]
tile_ids = ['tile0000']

for tile_idx, tile_id in enumerate(tile_ids):
    data_register_factory = DataRegistration(dataset_path=data_dir_path,
                                                overwrite_registered=False,
                                                perform_optical_flow=True,
                                                tile_idx=tile_idx)
    
    data_register_factory.load_registered_data(readouts=readout)
    data_register_factory._create_figure(readouts=readout)
    
    del data_register_factory
    gc.collect()