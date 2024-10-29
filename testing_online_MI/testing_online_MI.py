import numpy as np
import os
import json
import mne
import time
import pickle
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from mne.decoding import CSP

import pandas as pd

import subprocess

from src.UdpComms import UdpComms
from src.boards import setup_and_prepare_board
from src.processing import predict_one_trial_MI
from src.bids_files import save_raw_bids

# Import the parent directory and the src to system
actual_path = os.path.dirname(os.path.realpath(__file__))
parent_path = os.path.abspath(os.path.join(actual_path, os.pardir))
os.sys.path.append(parent_path)
os.sys.path.append(os.path.join(parent_path, 'src'))

# Import the path to save the data
from global_config import PATH_TO_SAVE_DATA_EEG_MI, PATH_TO_SAVE_MODELS_EEG_MI


########################## THIS INFORMATION MUST BE DEFINED EVERY TIME THAT YOU HAVE TO ADQUIRE DATA ##########################
# SELECT THE SUBJECT, SESSION, RUNS AND TASK TO TRAIN THE DECODING PIPELINE
info_eeg_online = {
        "subject_ID": "001",
        "session_ID": 0,
        "run_ID": 0,
        "task": "MI_testing_online", # Esto tampoco se tiene que modificar
        "project_name": "MIBCIproject",
        "gender": "femenine",
        "dominance": "right"
    }

info_to_load_model = {
    "training_subject_ID":  "001",
    "training_session_ID": "0",
    "training_task": "MI",
    "project_name": "MIBCIproject"
}


# Obtain the path of the script
script_folder = os.path.dirname(os.path.realpath(__file__))

#################### LOAD CONFIG FILES ####################
# Load the board configuration file
with open(os.path.join(script_folder, 'configs', 'board_config.json')) as file:
    board_config = json.load(file)  # Python dictionary

# Load the stimulation protocol configuration file
with open(os.path.join(script_folder, 'configs', 'stim_protocol_config.json')) as file:
    stim_protocol_config = json.load(file)  # Python dictionary

file_name = stim_protocol_config['file_name']
markers_dict = stim_protocol_config['markers_dict']
stim_protocol_file_path = os.path.join(script_folder, 'stimulation_protocol',stim_protocol_config['file_name'])

load_model_path = os.path.join(PATH_TO_SAVE_MODELS_EEG_MI,info_to_load_model['project_name'], 'sub-' + info_to_load_model['training_subject_ID'], 'ses-' + info_to_load_model['training_session_ID'])
   
# Load the trained decoding pipeline
with open(os.path.join(load_model_path,  'CSP_LDA.pkl.pkl'), 'rb') as file:
    trained_pipeline = pickle.load(file)


########################### SETUP AND START STREAMING ###########################
# Set up the board, prepare it and start streaming
board, mne_info, exg_channels = setup_and_prepare_board(board_config)
board.start_stream()

# Create UDP socket to use for receiving data from Unity application (necessary if a Unity application is used as stimulation protocol)
sock = UdpComms(udpIP="127.0.0.1", portTX=8000, portRX=8001, enableRX=True, suppressWarnings=False)
time.sleep(1.5)

# Execute the stimulation protocol file (in this case, a Unity application but it could be any other)
protocol = subprocess.Popen(stim_protocol_file_path)

markers_time_list = []
markers_code_list = []
trials_arrays_list = []
y_true_list = []
y_pred_list = []
trials_counter = 0

while True:   # Until the end_game marker is received
    # Check if new data has been received from the Unity application
    received_data = sock.ReceiveData()
    if received_data: # if NEW data has been received since last ReadReceivedData function call
        # Received data is a string with the format "marker_code-time stamp"
        markers_time_list.append(received_data)
        marker_code = int(received_data.split('-')[0])
        markers_code_list.append(marker_code)
        if marker_code != markers_dict['end_game']:
            # Insert marker in the board
            board.insert_marker(int(marker_code))
            if marker_code == markers_dict['start_trial']:
                trials_counter += 1
                print ("Trial number: " + str(trials_counter))   # Just to see the progress of the calibration in the console
            elif marker_code in [markers_dict['go_cue_MI'], markers_dict['go_cue_rest']]:
                trial_array, y_pred = predict_one_trial(board, exg_channels, mne_info, trained_pipeline)
                if marker_code == markers_dict['go_cue_rest']:
                    y_true = 0
                elif marker_code == markers_dict['go_cue_MI']:
                    y_true = 1
                
                if y_true == y_pred:
                    # This line is to send a marker to the Unity application
                    # Acá tenemos que eliminarla y reemplazarla por el código que controla el robotito
                    sock.SendData("5")
                    
                trials_arrays_list.append(trial_array)
                y_true_list.append(y_true)
                y_pred_list.append(y_pred)
        else:
            break


time.sleep(3)
protocol.kill()

# Get the full data from the board
data = board.get_board_data()
# Stop the board streaming
board.stop_stream()
board.release_session()

# Save the markers and the data following BIDS format
save_raw_bids(data, exg_channels, markers_dict, mne_info, info_eeg_online, PATH_TO_SAVE_DATA_EEG_MI, session_type='MI_testing_online')