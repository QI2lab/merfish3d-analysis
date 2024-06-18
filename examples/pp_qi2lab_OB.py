from wf_merfish.postprocess.postprocess import postprocess
from pathlib import Path

if __name__ == '__main__':
    
    dataset_path = Path('/mnt/data/qi2lab/20240317_OB_MERFISH_7')
    codebook_path = dataset_path / ('codebook.csv')
    bit_order_path = dataset_path / ('bit_order.csv')
    noise_map_path = Path('/home/qi2lab/Documents/github/wf-merfish/hot_pixel_image.tif')

    camera = 'flir'
    write_raw_camera_data = False
    write_polyDT_tiff = False
    run_hotpixel_correction = True
    run_shading_correction = False
    run_tile_registration = False
    run_global_registration =  False
    global_registration_parameters = {'data_to_fuse': 'polyDT', # 'all' or 'polyDT'
                                    'parallel_fusion': True}
    write_fused_zarr = False
    run_cellpose = False
    cellpose_parameters = {'diam_mean_pixels': 30,
                            'flow_threshold': 0.6,
                            'normalization': [10,80]}
    run_tile_decoding = True
    tile_decoding_parameters = {'normalization': [.1,80],
                                'calculate_normalization': True,
                                'exp_type': '3D',
                                'merfish_bits': 16,
                                'lowpass_sigma': (3,1,1),
                                'distance_threshold': 0.8,
                                'magnitude_threshold': 0.3,
                                'minimum_pixels': 27,
                                'fdr_target': .05}
    # smfish_parameters = {'bits': [17,18], 'threshold': -1}
    run_baysor = False
    baysor_parameters = {'baysor_path' : "/home/mabbasi/Baysor/bin/baysor/bin/./baysor",
                        'num_threads': 24,
                        'cell_size_microns': 10,
                        'min_molecules_per_cell': 20,
                        'cellpose_prior_confidence': 0.5}
    baysor_ignore_genes = False
    # baysor_genes_to_exclude = []
    baysor_filtering_parameters = {'cell_area_microns' : 7.5,
                                'confidence_cutoff' : 0.7,
                                'lifespan' : 100}
    run_mtx_creation = False
    mtx_creation_parameters = {'confidence_cutoff' : 0.7}

    func = postprocess(dataset_path = dataset_path, 
                       codebook_path = codebook_path,
                       bit_order_path = bit_order_path,
                       camera = camera,
                       write_raw_camera_data = write_raw_camera_data,
                       run_hotpixel_correction = run_hotpixel_correction,
                       run_shading_correction = run_shading_correction,
                       run_tile_registration = run_tile_registration,
                       write_polyDT_tiff = write_polyDT_tiff,
                       run_global_registration =  run_global_registration,
                       global_registration_parameters = global_registration_parameters,
                       run_cellpose = run_cellpose,
                       cellpose_parameters = cellpose_parameters,
                       run_tile_decoding = run_tile_decoding,
                       tile_decoding_parameters = tile_decoding_parameters,
                       # smfish_parameters = smfish_parameters,
                       run_baysor = run_baysor,
                       baysor_parameters = baysor_parameters,
                       baysor_ignore_genes = baysor_ignore_genes,
                       # baysor_genes_to_exclude = baysor_genes_to_exclude,
                       baysor_filtering_parameters = baysor_filtering_parameters,
                       run_mtx_creation = run_mtx_creation,
                       mtx_creation_parameters = mtx_creation_parameters,
                       noise_map_path = noise_map_path)
    
    for val in func:
        temp_val = val