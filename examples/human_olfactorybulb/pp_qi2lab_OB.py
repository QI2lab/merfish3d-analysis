from wf_merfish.postprocess.postprocess import postprocess
from pathlib import Path

if __name__ == '__main__':
    
    dataset_path = Path('/mnt/data/qi2lab/20240317_OB_MERFISH_7')
    codebook_path = dataset_path / ('codebook.csv')
    bit_order_path = dataset_path / ('bit_order.csv')
    noise_map_path = Path('/home/qi2lab/Documents/github/wf-merfish/hot_pixel_image.tif')

    camera = 'flir'
    write_raw_camera_data = True
    write_polyDT_tiff = False
    run_hotpixel_correction = True
    run_shading_correction = False
    run_tile_registration = True
    run_global_registration =  True
    global_registration_parameters = {'data_to_fuse': 'polyDT', # 'all' or 'polyDT'
                                    'parallel_fusion': True}
    run_cellpose = True
    cellpose_parameters = {'diam_mean_pixels': 30,
                            'flow_threshold': 0.6,
                            'normalization': [10,80]}
    run_tile_decoding = True
    tile_decoding_parameters = {'exp_type': '3D',
                                'minimum_pixels': 27,
                                'fdr_target': .15}
    # smfish_parameters = {'bits': [17,18], 'threshold': -1}
    run_baysor = True
    baysor_parameters = {'baysor_path' : "/home/qi2lab/Documents/github/Baysor/bin/baysor/bin/./baysor",
                        'num_threads': 24}
    baysor_ignore_genes = True
    baysor_genes_to_exclude = "Blank*"
    # baysor_genes_to_exclude = "OR10C1, OR10G2, OR10H1, OR10H5, OR10Q1, OR10S1, OR10W1, OR11A1,\
    #                         OR12D1, OR13A1, OR13J1, OR1F1, OR1I1, OR1M1, OR2A1, OR2A14,\
    #                         OR2A20P, OR2A4, OR2A42, OR2A9P, OR2AT4, OR2B11, OR2C1, OR2C3,\
    #                         OR2F1, OR2H1, OR2H2, OR2L13, OR2S2, OR2T2, OR2T27, OR2T35,\
    #                         OR2T5, OR2T7, OR2Z1, OR3A2, OR3A3, OR3A4P, OR51D1, OR51E1,\
    #                         OR51E2, OR51G1, OR52I1, OR52I2, OR52K2, OR52L1, OR52W1,\
    #                         OR56B1, OR56B4, OR5AU1, OR5C1, OR6A2, OR6J1, OR6W1P, OR7A5,\
    #                         OR8A1, OR9Q1, Blank*"
    run_mtx_creation = True
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
                       run_baysor = run_baysor,
                       baysor_parameters = baysor_parameters,
                       baysor_ignore_genes = baysor_ignore_genes,
                       baysor_genes_to_exclude=baysor_genes_to_exclude,
                       run_mtx_creation = run_mtx_creation,
                       mtx_creation_parameters = mtx_creation_parameters,
                       noise_map_path = noise_map_path)
    
    for val in func:
        temp_val = val