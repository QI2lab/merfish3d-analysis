import napari
from pathlib import Path
import zarr
import numpy as np
import gc
from qtpy.QtWidgets import QApplication

def on_close_callback():
    viewer.layers.clear()
    gc.collect()

data_dir_path = Path('/mnt/opm3/20240124_OB_Full_MERFISH_UA_3_allrds/processed_v2')

tile_id = 'tile0000'

readout_dir_path = data_dir_path / Path('readouts')
tile_dir_path = readout_dir_path / Path(tile_id)

bit_ids = [entry.name for entry in tile_dir_path.iterdir() if entry.is_dir()]

for bit_id in bit_ids:

    bit_dir_path = tile_dir_path / Path(bit_id)
    current_channel = zarr.open(bit_dir_path,mode='r')
    
    path_save_dir = data_dir_path / Path('localizations') / Path(tile_id) / Path(bit_id).stem
    path_save_dir.mkdir(parents=True, exist_ok=True) 
    path_localization_file = path_save_dir / Path ('localization_parameters.json')
    
    
    if not(path_localization_file.exists()):

        voxel_zyx_um = np.asarray(current_channel.attrs['voxel_zyx_um']).astype(np.float32)
        em_wvl = float(current_channel.attrs['emission_um'])
        data = np.asarray(current_channel['raw_data']).astype(np.uint16)
        viewer = napari.Viewer()
        app = QApplication.instance()

        app.lastWindowClosed.connect(on_close_callback)
        viewer.window._qt_window.setWindowTitle(tile_id + '; ' + bit_id + '; round'+ str(current_channel.attrs['round']).zfill(2))
        viewer.add_image(data)

        dock_widget, plugin_widget = viewer.window.add_plugin_dock_widget("napari-spot-detection", "Spot detection")

        plugin_widget.scan_chunk_size_deconv = 128
        plugin_widget.scan_chunk_size_dog = 16
        plugin_widget.scan_chunk_size_find_peaks = 128
                

        plugin_widget.txt_ri.setText('1.51')
        plugin_widget.txt_lambda_em.setText(str(em_wvl*1000))
        plugin_widget.txt_dc.setText(str(np.round(voxel_zyx_um[1]*2,3)))
        plugin_widget.txt_dstage.setText(str(voxel_zyx_um[0]))
        plugin_widget.chk_skewed.setChecked(True)
        plugin_widget.chk_skewed.setChecked(False)

        plugin_widget.path_save = path_localization_file
        plugin_widget.but_make_psf.click()
        plugin_widget.but_load_model.click()
        plugin_widget.txt_deconv_iter.setText('20')
        plugin_widget.but_run_deconvolution.click()
        plugin_widget.cbx_dog_choice.setCurrentIndex(0)
        plugin_widget.but_dog.click()
        plugin_widget.cbx_find_peaks_source.setCurrentIndex(0)
        viewer.reset_view()

        napari.run()
        

        del dock_widget, plugin_widget
        del viewer
        del data, voxel_zyx_um, em_wvl
        gc.collect()
        
import json

        
def parse_localization_parameters(path_localization_params_file: Path):
    
    with open(path_localization_params_file, "r") as read_file:
        detection_parameters = json.load(read_file)
            
        # parse all keys and modify the widgets' values
        metadata = {'pixel_size': float(detection_parameters['metadata']['pixel_size']),
                    'scan_step': float(detection_parameters['metadata']['scan_step']),
                    'wvl': float(detection_parameters['metadata']['wvl'])}
        
        microscope_params = {'na': float(detection_parameters['microscope_params']['na']),
                            'ri': float(detection_parameters['microscope_params']['ri']),
                            'theta': float(detection_parameters['microscope_params']['theta'])}
        
        decon_params = {'iterations': int(detection_parameters['decon_params']['iterations']),
                        'tv_tau': float(detection_parameters['decon_params']['tv_tau'])}
        
        DoG_filter_params = {'sigma_small_z_factor': float(detection_parameters['DoG_filter_params']['sigma_small_z_factor']),
                            'sigma_large_z_factor': float(detection_parameters['DoG_filter_params']['sigma_large_z_factor']),
                            'sigma_small_y_factor': float(detection_parameters['DoG_filter_params']['sigma_small_y_factor']),
                            'sigma_large_y_factor': float(detection_parameters['DoG_filter_params']['sigma_large_y_factor']),
                            'sigma_small_x_factor': float(detection_parameters['DoG_filter_params']['sigma_small_x_factor']),
                            'sigma_large_x_factor': float(detection_parameters['DoG_filter_params']['sigma_large_x_factor'])}
        
        find_candidates_params = {'threshold': float(detection_parameters['find_candidates_params']['threshold']),
                             'min_spot_z_factor': float(detection_parameters['find_candidates_params']['min_spot_z_factor']),
                             'min_spot_xy_factor': float(detection_parameters['find_candidates_params']['min_spot_xy_factor'])}
        
        fit_candidate_spots_params = {'n_spots_to_fit' : int(detection_parameters['fit_candidate_spots_params']['n_spots_to_fit']),
                                      'roi_z_factor' : float(detection_parameters['fit_candidate_spots_params']['roi_z_factor']),
                                      'roi_y_factor' : float(detection_parameters['fit_candidate_spots_params']['roi_y_factor']),
                                      'roi_x_factor' : float(detection_parameters['fit_candidate_spots_params']['roi_x_factor'])}
        
        spot_filter_params = {'sigma_min_xy_factor' : float(detection_parameters['spot_filter_params']['sigma_min_xy_factor']),
                              'sigma_max_xy_factor' : float(detection_parameters['spot_filter_params']['sigma_max_xy_factor']),
                              'sigma_min_z_factor' : float(detection_parameters['spot_filter_params']['sigma_min_z_factor']),
                              'sigma_max_z_factor' : float(detection_parameters['spot_filter_params']['sigma_max_z_factor']),
                              'min_sigma_ratio' : float(detection_parameters['spot_filter_params']['min_sigma_ratio']),
                              'max_sigma_ratio' : float(detection_parameters['spot_filter_params']['max_sigma_ratio']),
                              'amp_min' : float(detection_parameters['spot_filter_params']['amp_min']),
                              'amp_max' : float(detection_parameters['spot_filter_params']['amp_min']*50),
                              'fit_dist_max_err_z_factor' : float(detection_parameters['spot_filter_params']['fit_dist_max_err_z_factor']),
                              'fit_dist_max_err_xy_factor' : float(detection_parameters['spot_filter_params']['fit_dist_max_err_xy_factor']),
                              'min_spot_sep_z_factor' : float(detection_parameters['spot_filter_params']['min_spot_sep_z_factor']), 
                              'min_spot_sep_xy_factor' : float(detection_parameters['spot_filter_params']['min_spot_sep_xy_factor']),
                              'dist_boundary_z_factor' : float(detection_parameters['spot_filter_params']['dist_boundary_z_factor']),
                              'dist_boundary_xy_factor': float(detection_parameters['spot_filter_params']['dist_boundary_xy_factor'])}
        
            
        chained = {'deconvolve' : True,
                  'dog_filter' : True,
                  'find_candidates' : True,
                  'merge_candidates' : True,
                  'localize' : True,
                  'save' : True}
        
    return microscope_params, metadata, decon_params, DoG_filter_params, find_candidates_params, fit_candidate_spots_params, spot_filter_params, chained
        
from spots3d.SPOTS3D import SPOTS3D
from psfmodels import make_psf

data_dir_path = Path('/mnt/opm3/20240119_OB_Full_MERFISH_UA_2_allrds/processed_v2')

readout_dir_path = data_dir_path / Path('readouts')

tile_ids = [entry.name for entry in readout_dir_path.iterdir() if entry.is_dir()]

tile_dir_path = readout_dir_path / Path(tile_ids[0])

bit_ids = [entry.name for entry in tile_dir_path.iterdir() if entry.is_dir()]

for bit_id in bit_ids:
        
    path_localization_params_file = data_dir_path / Path('localizations') / Path(tile_ids[0]) / Path(bit_id).stem / Path ('localization_parameters.json')

    microscope_params, metadata, decon_params, DoG_filter_params,\
        find_candidates_params, fit_candidate_spots_params, spot_filter_params,\
        chained = parse_localization_parameters(path_localization_params_file)
        
    psf = make_psf(z=9,
            nx=15,
            dxy=metadata['pixel_size'],
            dz=metadata['scan_step'],
            NA=1.35,
            wvl=metadata['wvl'],
            ns=1.33,
            ni=1.51,
            ni0=1.51,
            model='vectorial')
        
    for tile_id in tile_ids:
        
        print(bit_id,tile_id)
        
        tile_dir_path = readout_dir_path / Path(tile_id)
        bit_dir_path = tile_dir_path / Path(bit_id)
        current_channel = zarr.open(bit_dir_path,mode='r')
        
        path_save_dir = data_dir_path / Path('localizations') / Path(tile_id) / Path(bit_id).stem
        path_save_dir.mkdir(parents=True, exist_ok=True) 
        base_name = 'localization_tile_coords'
        
        test_path = path_save_dir / Path ('localized_spots_localization_tile_coords.parquet')
        
        if not(test_path.exists()):
            
            data = np.asarray(current_channel['raw_data']).astype(np.uint16)
             
            spots3d = SPOTS3D(data=data,
                            psf=psf,
                            metadata=metadata,
                            microscope_params=microscope_params,
                            scan_chunk_size=128,
                            decon_params=decon_params,
                            DoG_filter_params=DoG_filter_params,
                            find_candidates_params=find_candidates_params,
                            fit_candidate_spots_params=fit_candidate_spots_params,
                            spot_filter_params=spot_filter_params,
                            chained=chained)
                            
            spots3d.dog_filter_source_data = 'decon'
            spots3d.find_candidates_source_data = 'dog'
            
            spots3d.run_deconvolution()

            spots3d.scan_chunk_size = 16
            spots3d.run_DoG_filter()
            spots3d.scan_chunk_size = 128
            spots3d.run_find_candidates()
            spots3d.run_fit_candidates()
            spots3d.run_filter_spots(return_values=True)
            spots3d.save_results(dir_localize=path_save_dir,
                                base_name=base_name)

            del data, spots3d
        else:
            pass
        
        
print('finished.')