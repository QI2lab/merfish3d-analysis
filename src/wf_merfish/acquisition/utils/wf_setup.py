from pathlib import Path
import numpy as np

def retrieve_setup_from_MM(core,studio,df_config,debug=False):
    """
    Parse MM GUI to retrieve exposure time, channels to use, powers, and stage positions

    :param core: Core
        active pycromanager MMcore object
    :param core: Studio
        active pycromanager MMstudio object
    :param df_config: dict
        dictonary containing instrument setup information
    :param debug: boolean
        flag to bring debug information
    
    :return df_MM_setup: dict
        dictonary containing scan configuration settings from MM GUI
    """

    # pull current MDA window settings
    acq_manager = studio.acquisitions()
    acq_settings = acq_manager.get_acquisition_settings()

    # grab settings from MM
    # grab and setup save directory and filename
    save_directory=Path(acq_settings.root())
    save_name=Path(acq_settings.prefix())

    # pull active LEDs from MDA window
    channel_labels = ['T-P1','T-P2','T-P3','T-P4','F-Blue', 'F-Yellow', 'F-Red']
    channel_states = [False,False,False,False,False,False,False] #define array to keep active channels
    channel_exposures = [53.0,53.0,53.0,53.0,0.,0.,0.]
    channels = acq_settings.channels() # get active channels in MDA window
    for idx in range(channels.size()):
        channel = channels.get(idx) # pull channel information
        if channel.config() == channel_labels[0]: 
            channel_states[0]=True
            channel_exposures[0]=53.0
        elif channel.config() == channel_labels[1]: 
            channel_states[1]=True
            channel_exposures[1]=53.0
        elif channel.config() == channel_labels[2]: 
            channel_states[2]=True
            channel_exposures[2]=53.0
        elif channel.config() == channel_labels[3]: 
            channel_states[3]=True
            channel_exposures[3]=53.0
        elif channel.config() == channel_labels[4]: 
            channel_states[4]=True
            channel_exposures[4]=channel.exposure()
        elif channel.config() == channel_labels[5]: 
            channel_states[5]=True
            channel_exposures[5]=channel.exposure()
        elif channel.config() == channel_labels[6]: 
            channel_states[6]=True
            channel_exposures[6]=channel.exposure()

    # calculate number of active channels
    n_active_channels = np.array(channel_states).sum()

    # set up XY positions
    position_list_manager = studio.positions()
    position_list = position_list_manager.get_position_list()
    number_positions = position_list.get_number_of_positions()
    x_positions = np.empty(number_positions)
    y_positions = np.empty(number_positions)
    z_positions = np.empty(number_positions)

    # iterate through position list to extract XY positions    
    for idx in range(number_positions):
        pos = position_list.get_position(idx)
        for ipos in range(pos.size()):
            stage_pos = pos.get(ipos)
            if (stage_pos.get_stage_device_label() == 'XYStage'):
                x_positions[idx] = stage_pos.x
                y_positions[idx] = stage_pos.y
            if (stage_pos.get_stage_device_label() == 'ZStage'):
                z_positions[idx] = stage_pos.x

    # grab Z stack information
    z_relative_start = acq_settings.slice_z_bottom_um()
    z_relative_end = acq_settings.slice_z_top_um()
    z_relative_step = acq_settings.slice_z_step_um()
    number_z_positions = int(np.abs(z_relative_end-z_relative_start)/z_relative_step + 1)

    # populate XYZ array
    xyz_positions = np.empty([number_positions,3])
    xyz_positions[:,0]= x_positions
    xyz_positions[:,1]= y_positions
    xyz_positions[:,2]= z_positions

    # grab autofocus information
    autofocus_manager = studio.get_autofocus_manager()
    # autofocus_method = autofocus_manager.get_autofocus_method()
    # autofocus_method.full_focus()
    # TO DO: need to get autofocus to respect delay. may need to change channel, snap a picture, then run autofocus

    # # set pixel size
    # pixel_size_um = float(df_config['pixel_size_um']) # unit: um 

    # determine image size
    # core.snap_image()
    # y_pixels = core.get_image_height()
    # x_pixels = core.get_image_width()

    # generate dictionary to return with scan parameters
    df_MM_setup = {'tile_positions': int(number_positions),
                    'axial_positions': int(number_z_positions),
                    'z_relative_start': float(z_relative_start),
                    'z_relative_end': float(z_relative_end),
                    'z_relative_step': float(z_relative_step),
                    'n_active_channels': int(n_active_channels),
                    'save_directory': str(save_directory),
                    'save_name': str(save_name),
                    'dpc1_active' : bool(channel_states[0]),
                    'dpc2_active' : bool(channel_states[1]), 
                    'dpc3_active' : bool(channel_states[2]),
                    'dpc4_active' : bool(channel_states[3]), 
                    'blue_active': bool(channel_states[4]),
                    'yellow_active': bool(channel_states[5]),
                    'red_active': bool(channel_states[6]),
                    'dpc1_exposure': float(channel_exposures[0]),
                    'dpc2_exposure': float(channel_exposures[1]),
                    'dpc3_exposure': float(channel_exposures[2]),
                    'dpc4_exposure': float(channel_exposures[3]),
                    'blue_exposure': float(channel_exposures[4]),
                    'yellow_exposure': float(channel_exposures[5]),
                    'red_exposure': float(channel_exposures[6])}

    return df_MM_setup, xyz_positions, autofocus_manager