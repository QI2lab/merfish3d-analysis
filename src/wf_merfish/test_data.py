from DataRegistration import DataRegistration
from pathlib import Path

data_path = Path('/mnt/opm3/20231108_pc_ampvsun_v2/processed/raw_zarr/data.zarr')
codebook_path = Path('/mnt/opm3/20231017 BiFISH experiment full run/codebook_files/codebook 20231017 rounds copy.txt')
exporder_path = Path('/mnt/opm3/20231017 BiFISH experiment full run/codebook_files/codebook 202301017 genes and barcodes.txt')

data_register_factory = DataRegistration(dataset_path=data_path,
                                        codebook_path=None,
                                        exp_order_path=None,
                                        overwrite_registered=False,
                                        overwrite_hotpixel=False,
                                        reference_channel='F-Red',
                                        selected_channels={'DPC': True,
                                                        'F-Blue': False,
                                                        'F-Yellow': False,
                                                        'F-Red': True},
                                        tile_idx=0)

dataset_metadata = data_register_factory.dataset_metadata

for tile_idx in range(0,dataset_metadata['num_tiles']):
    print('Tile: '+str(tile_idx).zfill(3))
    print('---------')
    data_register_factory.tile_idx = tile_idx
    data_register_factory._has_hotpixel_corrected=True
    data_register_factory._overwrite_registered=True
    data_register_factory.generate_registrations()
    data_register_factory.load_rigid_registrations()
    data_register_factory.apply_registrations()