'''
ASU multi-channel widefield LED scope using pycromanager.

Shepherd 09/23 - new code based on OPM changes
'''

# imports
from hardware.APump import APump
from hardware.HamiltonMVP import HamiltonMVP

# qi2lab OPM stage scan control functions for pycromanager
from utils.data_io import read_config_file, read_fluidics_program, write_metadata, time_stamp, append_index_filepath
from utils.fluidics_control import run_fluidic_program_wf
from utils.wf_setup import retrieve_setup_from_MM

# pycromanager
from pycromanager import Core, Acquisition, Studio

# python libraries
import time
import sys
import gc
from pathlib import Path
import easygui
from tqdm import tqdm
import numpy as np

def main():
    """"
    Execute iterative, interleaved OPM stage scan using MM GUI
    """
    # flags for resuming acquisition
    # TO DO: Make this a user setting during acquisition so that it can't screw up a restart.
    resume_tile = 0

    # flags for metadata, processing, drift correction
    setup_metadata = True
    debug_flag = True
    switch_last_round = False
    avoid_overwriting = True

    # check if user wants to flush system?
    run_fluidics = False
    flush_system = False
    run_type = easygui.choicebox('Type of run?', 'WF setup',
                                 ['Run fluidics (no imaging)', 'Iterative imaging', 'Single round (test)'])
    if run_type == str('Run fluidics (no imaging)'):
        flush_system = True
        run_fluidics = True
        # load fluidics program
        fluidics_path = easygui.fileopenbox('Load fluidics program')
        program_name = Path(fluidics_path)
    elif run_type == str('Iterative imaging'):
        flush_system = False
        run_fluidics = True
        # load fluidics program
        fluidics_path = easygui.fileopenbox('Load fluidics program')
        program_name = Path(fluidics_path)
    elif run_type == str('Single round (test)'):
        flush_system = False
        run_fluidics = False
        n_iterative_rounds = 1
        fluidics_rounds = [1]
        program_name = None

    file_directory = Path(__file__).resolve().parent
    config_file = file_directory / Path('wf_config.csv')
    df_config = read_config_file(config_file)

    if run_fluidics:
        # define ports for pumps and valves
        pump_COM_port = str(df_config['pump_com_port'])
        valve_COM_port = str(df_config['valve_com_port'])

        # setup pump parameters
        pump_parameters = {'pump_com_port': pump_COM_port,
                           'pump_ID': 30,
                           'verbose': True,
                           'simulate_pump': False,
                           'serial_verbose': False,
                           'flip_flow_direction': False}

        # connect to pump
        pump_controller = APump(pump_parameters)

        # set pump to remote control
        pump_controller.enableRemoteControl(True)

        # connect to valves
        valve_controller = HamiltonMVP(com_port=valve_COM_port)

        # initialize valves
        valve_controller.autoAddress()

        # load user defined program from hard disk
        df_program = read_fluidics_program(program_name)
        fluidics_rounds = df_program["round"].unique()
        print('Will execute fluidics rounds:', fluidics_rounds)
        # When resuming fluidics experiments, the max round name and number can differ
        max_round_name = df_program["round"].max()
        n_iterative_rounds = len(fluidics_rounds)
        print('Number of iterative rounds: ' + str(n_iterative_rounds))
        if max_round_name != n_iterative_rounds:
            print(f"Max round label is {max_round_name}")

    if flush_system:
        # run fluidics program for this round
        success_fluidics = False
        success_fluidics = run_fluidic_program_wf(1, df_program, valve_controller, pump_controller)
        if not (success_fluidics):
            print('Error in fluidics! Stopping scan.')
            sys.exit()
        print('Flushed fluidic system.')
        sys.exit()

    # connect to Micromanager core instance
    core = Core()

    # change core timeout for long stage moves
    core.set_property('Core', 'TimeoutMs', 500000)
    time.sleep(1)

    # iterate over user defined program
    for r_idx, r_name in enumerate(fluidics_rounds):

        # studio = bridge.get_studio()
        studio = Studio()

        # get handle to xy and z stages
        xy_stage = core.get_xy_stage_device()
        z_stage = core.get_focus_device()

        time_now = time.perf_counter()

        if run_fluidics:
            success_fluidics = False
            success_fluidics = run_fluidic_program_wf(r_name, df_program, valve_controller, pump_controller)
            if not (success_fluidics):
                print('Error in fluidics! Stopping scan.')
                sys.exit()

        # if first round, have user setup positions, laser intensities, and exposure time in MM GUI
        if r_idx == 0:

            # setup imaging parameters using MM GUI
            run_imaging = False
            while not (run_imaging):

                setup_done = False
                while not (setup_done):
                    setup_done = easygui.ynbox('Finished setting up MM?', 'Title', ('Yes', 'No'))

                df_MM_setup, xyz_positions, autofocus_manager  = retrieve_setup_from_MM(core, studio, df_config, debug=debug_flag)

                channel_states = [bool(df_MM_setup['dpc1_active']),
                                  bool(df_MM_setup['dpc2_active']),
                                  bool(df_MM_setup['dpc3_active']),
                                  bool(df_MM_setup['dpc4_active']),
                                  bool(df_MM_setup['blue_active']),
                                  bool(df_MM_setup['yellow_active']),
                                  bool(df_MM_setup['red_active'])]

                channel_exposures = [float(df_MM_setup['dpc1_exposure']),
                                     float(df_MM_setup['dpc2_exposure']),
                                     float(df_MM_setup['dpc3_exposure']),
                                     float(df_MM_setup['dpc4_exposure']),
                                     float(df_MM_setup['blue_exposure']),
                                     float(df_MM_setup['yellow_exposure']),
                                     float(df_MM_setup['red_exposure'])]

                # construct and display imaging summary to user
                scan_settings = (f"Number of labeling rounds: {str(n_iterative_rounds)} \n\n"
                                 f"Number of XY tiles:  {str(df_MM_setup['tile_positions'])} \n"
                                 f"Number of Z planes: {str(df_MM_setup['axial_positions'])} \n"
                                 f"Number of channels:  {str(df_MM_setup['n_active_channels'])} \n"
                                 f"Active LEDs: {str(channel_states)} \n"
                                 f"LED exposures: {str(channel_exposures)} \n")

                output = easygui.textbox(scan_settings, 'Please review scan settings')

                # verify user actually wants to run imaging
                run_imaging = easygui.ynbox('Run acquistion?', 'Title', ('Yes', 'No'))

        # if last round, switch to DAPI + alexa488 readout instead
        if switch_last_round and (r_idx == (n_iterative_rounds - 1)) and (run_fluidics):
   
            setup_done = False
            while not (setup_done):
                setup_done = easygui.ynbox('Finished setting up MM?', 'Title', ('Yes', 'No'))

            df_MM_setup, xyz_positions, autofocus_manager  = retrieve_setup_from_MM(core, studio, df_config, debug=debug_flag)

            channel_states = [bool(df_MM_setup['dpc1_active']),
                            bool(df_MM_setup['dpc2_active']),
                            bool(df_MM_setup['dpc3_active']),
                            bool(df_MM_setup['dpc4_active']),
                            bool(df_MM_setup['blue_active']),
                            bool(df_MM_setup['yellow_active']),
                            bool(df_MM_setup['red_active'])]

            channel_exposures = [float(df_MM_setup['dpc1_exposure']),
                                    float(df_MM_setup['dpc2_exposure']),
                                    float(df_MM_setup['dpc3_exposure']),
                                    float(df_MM_setup['dpc4_exposure']),
                                    float(df_MM_setup['blue_exposure']),
                                    float(df_MM_setup['yellow_exposure']),
                                    float(df_MM_setup['red_exposure'])]

            # construct and display imaging summary to user
            scan_settings = (f"Number of labeling rounds: {str(n_iterative_rounds)} \n\n"
                                f"Number of XY tiles:  {str(df_MM_setup['tile_positions'])} \n"
                                f"Number of Z planes: {str(df_MM_setup['axial_positions'])} \n"
                                f"Number of channels:  {str(df_MM_setup['n_active_channels'])} \n"
                                f"Active LEDs: {str(channel_states)} \n"
                                f"LED exposures: {str(channel_exposures)} \n")

            output = easygui.textbox(scan_settings, 'Please review scan settings')

            # verify user actually wants to run imaging
            run_imaging = easygui.ynbox('Run acquistion?', 'Title', ('Yes', 'No'))

        gc.collect()

        for tile_idx in tqdm(range(resume_tile,xyz_positions.shape[0])):

            # move XY stage to new tile axis position
            core.set_xy_position(xyz_positions[tile_idx,0],xyz_positions[tile_idx,1])
            core.wait_for_device(xy_stage)
            core.set_position(xyz_positions[tile_idx,2])
            core.wait_for_device(z_stage)

            # update save_name with current tile information
            save_name_rxyz = Path(
                str(df_MM_setup['save_name']) + '_r' + str(r_name).zfill(4) + '_tile' + str(tile_idx).zfill(4))
            if avoid_overwriting:
                save_name_rxyz = append_index_filepath(save_name_rxyz)

            # query current stage positions
            xy_pos = core.get_xy_stage_position()
            stage_x = np.round(float(xy_pos.x), 2)
            stage_y = np.round(float(xy_pos.y), 2)
            stage_z = np.round(float(core.get_position()), 2)

            possible_channels = ['T-P1','T-P2','T-P3','T-P4','F-Blue','F-Yellow','F-Red']
            channels_active = []
            channel_exposures_ms = []
            for idx, c in enumerate(possible_channels):
                if channel_states[idx]:
                    channels_active.append(possible_channels[idx])
                    channel_exposures_ms.append(channel_exposures[idx])

            # TO DO: run autofocus here and modify z values as needed
            # core.set_config('LED','F-Blue')
            # core.set_auto_shutter(False)
            # core.set_shutter_open(True)
            # autofocus_method = autofocus_manager.get_autofocus_method()
            # autofocus_method.full_focus()
            # core.set_shutter_open(False)
            # core.set_auto_shutter(True)
            # found_z = np.round(float(core.get_position()), 2)
            # offset_z = found_z - stage_z
            offset_z = 0

            # create stage position dictionary
            current_stage_data = [{'stage_x': float(stage_x),
                                    'stage_y': float(stage_y),
                                    'stage_z': float(stage_z),
                                    'offset_z': float(offset_z),
                                    'dpc1_active' : bool(channel_states[0]),
                                    'dpc2_active' : bool(channel_states[1]),
                                    'dpc3_active' : bool(channel_states[2]),
                                    'dpc4_active' : bool(channel_states[3]),
                                    'blue_active': bool(channel_states[4]),
                                    'yellow_active': bool(channel_states[5]),
                                    'red_active': bool(channel_states[6])}]

            z_start = float(df_MM_setup['z_relative_start'])
            z_end = float(df_MM_setup['z_relative_end'])
            z_step = float(df_MM_setup['z_relative_step'])

            x_planned = xyz_positions[tile_idx,0]
            y_planned = xyz_positions[tile_idx,1]
            z_planned = xyz_positions[tile_idx,2] #+ offset_z
            z_range = np.arange(z_planned-np.abs(z_start),z_planned+np.abs(z_end),z_step)

            events=[]
            for chan, exp_ms in zip(channels_active,channel_exposures_ms):
                for z_idx, z_pos in enumerate(z_range):
                    evt = { 'axes': {'z': z_idx, 'channel': chan},
                                'x': np.round(x_planned,2),
                                'y': np.round(y_planned,2),
                                'z': np.round(z_pos,2), 
                                'config_group': ['LED',chan],
                                'exposure': exp_ms,
                                'keep_shutter_open': True} # DPS: fix shutter state to avoid Arduino issue!
                    events.append(evt)
         
            # run acquisition for this rxyz combination
            print(time_stamp(),
                    f'round {r_idx + 1}/{n_iterative_rounds}; tile {tile_idx + 1}/{xyz_positions.shape[0]}.')
            print(time_stamp(), f'Stage location (um): x={stage_x}, y={stage_y}, z={stage_z}.')
            core.set_auto_shutter(False)
            core.set_shutter_open(True)
            with Acquisition(directory=str(df_MM_setup['save_directory']), name=str(save_name_rxyz),show_display=False,debug=False) as acq:
                acq.acquire(events)
            acq = None
            del acq
            gc.collect()
            core.set_shutter_open(False)
            core.set_auto_shutter(True)

            # save experimental info after first tile.
            # we do it this way so that Pycromanager can manage directory creation

            if (setup_metadata):
                # save stage scan parameters
                scan_param_data = [{'root_name': str(df_MM_setup['save_name']),
                                    'scan_type': str('WF-stage-v4'),
                                    'num_t': int(1),
                                    'num_r': int(n_iterative_rounds),
                                    'num_xyz': int(xyz_positions.shape[0]),
                                    'num_ch': int(df_MM_setup['n_active_channels']),
                                    'dpc1_active': bool(channel_states[0]),
                                    'dpc2_active': bool(channel_states[1]),
                                    'dpc3_active': bool(channel_states[2]),
                                    'dpc4_active': bool(channel_states[3]),
                                    'blue_active': bool(channel_states[4]),
                                    'yellow_active': bool(channel_states[5]),
                                    'red_active': bool(channel_states[6]),
                                    'dpc1_exposure': float(channel_exposures[0]),
                                    'dpc2_exposure': float(channel_exposures[1]),
                                    'dpc3_exposure': float(channel_exposures[2]),
                                    'dpc4_exposure': float(channel_exposures[3]),
                                    'blue_exposure': float(channel_exposures[4]),
                                    'yellow_exposure': float(channel_exposures[5]),
                                    'red_exposure': float(channel_exposures[6])}]
                scan_metadata_path = Path(df_MM_setup['save_directory']) / 'scan_metadata.csv'
                if avoid_overwriting:
                    scan_metadata_path = append_index_filepath(scan_metadata_path)
                write_metadata(scan_param_data[0], scan_metadata_path)

                setup_metadata = False

            # save stage scan positions after each tile
            save_name_stage_positions = Path(
                str(df_MM_setup['save_name']) + '_r' + str(r_name).zfill(4) + '_tile' + str(tile_idx).zfill(
                    4) + '_stage_positions.csv')
            save_name_stage_path = Path(df_MM_setup['save_directory']) / save_name_stage_positions
            if avoid_overwriting:
                save_name_stage_path = append_index_filepath(save_name_stage_path)
            write_metadata(current_stage_data[0], save_name_stage_path)

        resume_tile = 0

        # run empty acquisition to close file handle to previous acquistion
        # need to do this because we process and delete raw data on the fly
        events_flush = None
        with Acquisition(directory=None, name=None, show_display=False) as acq_flush:
            acq_flush.acquire(events_flush)
        acq_flush = None
        del acq_flush
        gc.collect()

    # shut down python initialized hardware
    if (run_fluidics):
        # shutter_controller.close()
        valve_controller.close()
        pump_controller.close()

    del core, studio
    gc.collect()

if __name__ == "__main__":
    main()