import json
import os
import time
import subprocess

from src.UdpComms import UdpComms
from src.boards import setup_and_prepare_board
from src.bids_files import save_raw_bids

script_path = os.path.dirname(os.path.realpath(__file__))
os.chdir(script_path)
root_path = os.path.dirname(script_path)

# Define the path to save the data
save_data_path = os.path.join(root_path, 'data')
if not os.path.exists(save_data_path):
    os.makedirs(save_data_path)

# Load the subject config
with open(os.path.join(root_path, 'configs', 'subject_config.json')) as file:
    subject_config = json.load(file)

# Load the experiment config
with open(os.path.join(root_path, 'configs', 'experiment_config.json')) as file:
    experiment_config = json.load(file)

# Load the board configuration file
with open(os.path.join(root_path, 'configs', 'board_config.json')) as file:
    board_config = json.load(file)  # Python dictionary

# Load the stimulation protocol configuration file
with open(os.path.join(root_path, 'configs', 'stim_protocol_config.json')) as file:
    stim_protocol_config = json.load(file)  # Python dictionary
folder_name = stim_protocol_config['calibration']['folder_name']
file_name = stim_protocol_config['calibration']['file_name']
markers_dict = stim_protocol_config['calibration']['markers_dict']
stim_protocol_file_path = os.path.join(root_path, 'stimulation protocols', folder_name, file_name)

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
            board.insert_marker(marker_code)
            if marker_code == markers_dict['start_trial']:
                trials_counter += 1
                print ("Trial number: " + str(trials_counter))   # Just to see the progress of the calibration in the console
        else:
            break

time.sleep(3)
# End the game
protocol.kill()

# Get the full data from the board
data = board.get_board_data()
# Stop the board streaming
board.stop_stream()
board.release_session()

# Save the markers and the data following BIDS format
save_raw_bids(data, exg_channels, markers_dict, mne_info, subject_config, save_data_path, experiment_config, session_type='calibration')