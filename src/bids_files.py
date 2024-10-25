import datetime
import mne
from mne_bids import write_raw_bids
from mne_bids import BIDSPath
import numpy as np
import os

def save_raw_bids(data, exg_channels, markers_dict, mne_info, info, save_data_path, session_type): 
    """
    Save the raw data in BIDS format.

    """
    
    # Create the MNE raw object
    eeg_data = data[exg_channels]
    eeg_data = eeg_data / 1000000  # BrainFlow returns data in uV, convert to V for MNE
    raw = mne.io.RawArray(eeg_data, mne_info)
    
    # Measurement date
    meas_date = datetime.datetime.now(datetime.timezone.utc)
    raw.set_meas_date(meas_date)
    raw.info['line_freq'] = 50
    
    # Gender
    gender = info['gender']
    if (gender == 'masculine'):
        gen = 1
    elif (gender == 'female'):
        gen = 2
    else:
        gen = 0
    
    # Dominance
    dominance = info['dominance']
    if (dominance == 'right'):
        domi = 1
    elif (dominance == 'left'):
        domi = 2
    else:
        domi = 3
    
    raw.info['subject_info'] = {'sex': gen, 'birthday': None, 'hand': domi}
    
    bids_path = BIDSPath(subject=info['subject_ID'],
                         session=info['session_ID'],
                         task=session_type,
                         run=info['run_ID'],
                         root=os.path.join(save_data_path, info['project_name']))
    
    # Get the events codes from the data
    events = data[-1][data[-1] != 0]
    # Get the events times
    events_times = np.where(data[-1] != 0)[0]
    # Create the MNE-like events array
    events_array = np.zeros((events.size, 3))
    events_array[:, 0] = events_times
    events_array[:, 2] = events

    # Save the raw data in BIDS format
    write_raw_bids(raw, bids_path, format='BrainVision', allow_preload=True, events=events_array, event_id=markers_dict, overwrite=True)
        
        
   