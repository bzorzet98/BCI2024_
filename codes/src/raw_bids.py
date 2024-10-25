import datetime

from brainflow import BoardIds
import os.path as op
import mne
from mne_bids import write_raw_bids
from mne_bids import BIDSPath
import json
import os
import numpy as np

def save_raw_bids(board_id, principal_board, secondary_board, data_principal, data_secondary, infor, markers_list, run, folder_path, type_exp):
    """
    Write BIDS format files.
    
    Parameters
    ----------
    board_id : int
        Integer that determines which board is used:
            Synthetic: -1
            Cyton: 0
            Ganglion: 1
            Cyton Daisy: 2
    board : board_shim.BoardShim
        brainflow.board_shim object
        allows to read the board.
    data : Array de float64
        Values from the board.
        Dimensions depending on the type and time of acquisition.

    Returns
    -------
    None.

    """
    
    # Guardo la info de la cyton + daisy
    eeg_channels = principal_board.get_eeg_channels(board_id)
    eeg_data = data_principal[eeg_channels, :]
    eeg_data = eeg_data / 1000000  # BrainFlow returns uV, convert to V for MNE

    # Creating MNE objects from brainflow data arrays
    ch_types = ['eeg'] * len(eeg_channels)
    # ch_names = principal_board.get_eeg_names(board_id)
    with open('ch_names.txt', 'r') as file:
        ch_names = [linea.strip() for linea in file.readlines()]
               
    sfreq = principal_board.get_sampling_rate(board_id)
    info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types=ch_types)
    raw = mne.io.RawArray(eeg_data, info)
    
    # measurement date
    meas_date = datetime.datetime.now(datetime.timezone.utc)
    raw.set_meas_date(meas_date)
    
    raw.info['line_freq']=50
    
    # Date of Birth
    #Error acá
    dob = infor[0]['Fecha_de_Nacimiento']
    year = int(dob[6:])
    month = int(dob[3:5])
    day = int(dob[0:2])
    # dob = [year,month,day]
    dob = datetime.date(year, month, day)
    
    # gender
    gender = infor[0]['Genero']
    if (gender == 'Masculino'):
        gen = 1
    elif (gender == 'Femenino'):
        gen = 2
    else:
        gen = 0
    
    # dominance
    dominance = infor[0]['Dominancia']
    if (dominance == 'Derecha'):
        domi = 1
    elif (dominance == 'Izquierda'):
        domi = 2
    else:
        domi = 3
    
    # raw.info['subject_info']={'sex':gen,'birthday':dob,'hand':domi}
    raw.info['subject_info']={'sex':gen,'birthday':None,'hand':domi}
    data_path = folder_path + '/' + infor[0]['Tarea'] + '/BIDS/'
    
    bids_path = BIDSPath(subject=infor[0]['Sujeto'], session=infor[0]['Sesion'],
                         task=infor[0]['Tarea'], run='0'+str(run), root=data_path)
    
    eventos = []
    tam = len(ch_names)+15 
    for i in range(len(data_principal[tam])):
        if (data_principal[tam][i]!=0):
            eventos.append([i,0, data_principal[tam][i]])

            
    eventos_array=np.array(eventos[:][:][:])
    if type_exp == 'artifacts':
        bids_path = BIDSPath(subject=infor[0]['Sujeto'], session='calibration',
                             task='artifacts', root=data_path)
    elif type_exp == 'EEGbasal':
        if infor[0]['Sesion'] != '0':
            bids_path = BIDSPath(subject=infor[0]['Sujeto'], session='0' + infor[0]['Sesion'],
                                task='EEGbasal', root=data_path)
        else: 
            bids_path = BIDSPath(subject=infor[0]['Sujeto'], session='calibration',
                                task='EEGbasal', root=data_path)

    elif type_exp == 'preexperiment':
        bids_path = BIDSPath(subject=infor[0]['Sujeto'], session='calibration',
                             task='preexperiment', root=data_path)
    elif type_exp == 'calibration':
        bids_path = BIDSPath(subject=infor[0]['Sujeto'], session='calibration',
                             task='calibration', run='0' + str(run), root=data_path)
    elif type_exp == 'recalibration':
        bids_path = BIDSPath(subject=infor[0]['Sujeto'], session='0' + infor[0]['Sesion'],
                             task='recalibration', root=data_path)
    elif type_exp == 'closedloop':
        bids_path = BIDSPath(subject=infor[0]['Sujeto'], session='0' + infor[0]['Sesion'],
                             task='closedloop', run='0' + str(run), root=data_path)

    write_raw_bids(raw, bids_path, format='BrainVision', allow_preload=True, events=eventos_array, event_id=markers_list, overwrite=True);        
    
    # events.json           
    fileName = data_path + 'task-'+ infor[0]['Tarea'] + '_events.json'    
    data = {}
    data['onset'] = []
    data['onset'].append({
        'Description': 'Event onset',
        'Units': 'second'})
    
    data['duration'] = []
    data['duration'].append({
        'Description': 'Event duration',
        'Units': 'second'})
    
    data['value'] = []
    data['value'].append({
        'Description': 'Value of event (numerical)',
        'Levels': {str(value): key for key, value in markers_list.items()}})
    with open(os.path.join(fileName), 'w') as file:
        json.dump(data, file, indent=4)
        
        
    ##########################################################################3    
    # Guardo la info de la ganglion
    ch_emg_names = ['EMG 1', 'EMG 2', 'EMG 3', 'EMG 4','NA1', 'NA2', 'NA3', 'NA4','NA5', 'NA6', 'NA7', 'NA8','NA9', 'NA10', 'NA11']
    ch_emg_type = ['emg']*15
    emg_data = mne.create_info(ch_names=ch_emg_names, sfreq=BoardIds.GANGLION_NATIVE_BOARD, ch_types=ch_emg_type)
    raw_emg = mne.io.RawArray(data_secondary, emg_data)
    
    # measurement date
    meas_date = datetime.datetime.now(datetime.timezone.utc)
    raw_emg.set_meas_date(meas_date)
    
    raw_emg.info['line_freq']=50
    
    # Date of Birth
    #Error acá
    dob = infor[0]['Fecha_de_Nacimiento']
    year = int(dob[6:])
    month = int(dob[3:5])
    day = int(dob[0:2])
    dob = [year,month,day]
    
    # gender
    gender = infor[0]['Genero']
    if (gender == 'Masculino'):
        gen = 1
    elif (gender == 'Femenino'):
        gen = 2
    else:
        gen = 0
    
    # dominance
    dominance = infor[0]['Dominancia']
    if (dominance == 'Derecha'):
        domi = 1
    elif (dominance == 'Izquierda'):
        domi = 2
    else:
        domi = 3
    
    raw_emg.info['subject_info']={'sex':gen,'birthday':None,'hand':domi}
    
    if type_exp == 'artifacts':
        bids_path = folder_path + '/' + infor[0]['Tarea'] + '/BIDS/sub-' + infor[0]['Sujeto'] + '/ses-calibration/emg/'
        file_name = 'sub-' + infor[0]['Sujeto'] + '_ses-calibration_task-artifacts_emg'
    elif type_exp == 'preexperiment':
        bids_path = folder_path + '/' + infor[0]['Tarea'] + '/BIDS/sub-' + infor[0]['Sujeto'] + '/ses-calibration/emg/'
        file_name = 'sub-' + infor[0]['Sujeto'] + '_ses-calibration_task-preexperiment_emg'
    elif type_exp == 'EEGbasal':
        if infor[0]['Sesion'] == '0':
            bids_path = folder_path + '/' + infor[0]['Tarea'] + '/BIDS/sub-' + infor[0]['Sujeto'] + '/ses-calibration/emg/'
            file_name = 'sub-' + infor[0]['Sujeto'] + '_ses-calibration_task-EEGbasal_emg'
        else:
            bids_path = folder_path + '/' + infor[0]['Tarea'] + '/BIDS/sub-' + infor[0]['Sujeto'] + '/ses-0' + infor[0]['Sesion'] + '/emg/'
            file_name = 'sub-' + infor[0]['Sujeto'] + '_ses-0' + infor[0]['Sesion'] + '_task-EEGbasal_emg'
    elif type_exp == 'recalibration':
        bids_path = folder_path + '/' + infor[0]['Tarea'] + '/BIDS/sub-' + infor[0]['Sujeto'] + '/ses-0' + infor[0]['Sesion'] + '/emg/'
        file_name = 'sub-' + infor[0]['Sujeto'] + '_ses-0' + infor[0]['Sesion'] + '_task-recalibration_emg'
    elif type_exp == 'calibration':
        bids_path = folder_path + '/' + infor[0]['Tarea'] + '/BIDS/sub-' + infor[0]['Sujeto'] + '/ses-calibration/emg/'
        file_name = 'sub-' + infor[0]['Sujeto'] + '_ses-0' + infor[0]['Sesion'] + '_task-calibration_run-0' + str(run) + '_emg'
    elif type_exp == 'closedloop':
        bids_path = folder_path + '/' + infor[0]['Tarea'] + '/BIDS/sub-' + infor[0]['Sujeto'] + '/ses-0' + infor[0]['Sesion'] + '/emg/'
        file_name = 'sub-' + infor[0]['Sujeto'] + '_ses-0' + infor[0]['Sesion'] + '_task-closedloop_run-0' + str(run) + '_emg'

    full_path_emg = os.path.join(bids_path, file_name + '.fif')
    os.makedirs(bids_path, exist_ok=True)
    raw_emg.save(full_path_emg, overwrite=True)
    
    # events.json           
    fileName = data_path + 'task-'+ infor[0]['Tarea'] + '_run-' + str(run) + '_events.json'    
    data = {}
    data['onset'] = []
    data['onset'].append({
        'Description': 'Event onset',
        'Units': 'second'})
    
    data['duration'] = []
    data['duration'].append({
        'Description': 'Event duration',
        'Units': 'second'})
    
    data['value'] = []
    data['value'].append({
        'Description': 'Value of event (numerical)',
        'Levels': {str(value): key for key, value in markers_list.items()}})
    with open(os.path.join(fileName), 'w') as file:
        json.dump(data, file, indent=4)
        
        