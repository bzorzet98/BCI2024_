import numpy as np
import os
import json
import mne

import pickle
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from mne.decoding import CSP

import pandas as pd


# Import the parent directory and the src to system
actual_path = os.path.dirname(os.path.realpath(__file__))
parent_path = os.path.abspath(os.path.join(actual_path, os.pardir))
os.sys.path.append(parent_path)
os.sys.path.append(os.path.join(parent_path, 'src'))

# Import the path to save the data
from global_config import PATH_TO_SAVE_DATA_EEG_MI, PATH_TO_SAVE_MODELS_EEG_MI

mne.set_log_level(verbose='warning') #to avoid info at terminal

# SELECT THE SUBJECT, SESSION, RUNS AND TASK TO TRAIN THE DECODING PIPELINE
training_subject_ID =  "001"
training_session_ID = "0",
training_runs_ID = ["1"],
training_task = "MI",
project_name = "MIBCIproject"

epochs = []
for run in training_runs_ID:
    # Load the training data
    training_folder_path = os.path.join(PATH_TO_SAVE_DATA_EEG_MI, project_name, 'sub-' + training_subject_ID, 'ses-' + training_session_ID, 'eeg')
    filename = os.path.join(training_folder_path, 'sub-' + training_subject_ID + '_ses-' + training_session_ID + '_task-' + training_task + '_run-' + run + '_eeg.vhdr')
    raw = mne.io.read_raw_brainvision(filename, preload=True)

    # Drop the last channel, since it is not being recorded
    raw.drop_channels('NA')
    
    # Create events from annotations
    event_annot = pd.read_csv(filename[:-8] + 'events.tsv', sep='\t')
    event_annot = event_annot.loc[event_annot['trial_type'].isin(['go_cue_MI', 'go_cue_rest'])]
    events_matrix = np.vstack((event_annot['sample'],
                                event_annot['duration'],
                                event_annot['value'])).T
    events_matrix = events_matrix.astype('int32')
    # Set montage
    ten_twenty_montage = mne.channels.make_standard_montage(
        'standard_1020')
    raw.info.set_montage(ten_twenty_montage)

    ########################### PREPROCESSING ###########################
    # Band-pass and notch filtering
    raw.filter(1, 60)
    raw.notch_filter(50)

    # Epoching
    tmin, tmax = -3, 6  # -1 s before trial indicative
    event_ids = dict(MI=421, rest=422)  # map event IDs to tasks

    epochs_i = mne.Epochs(raw, events_matrix, tmin=tmin, tmax=tmax,
                         event_id=event_ids,
                         reject=None, baseline=None,
                         preload=True)
    
    epochs_i.filter(8, 30) # mu and beta band

    epochs.append(epochs_i)

epochs = mne.concatenate_epochs(epochs, verbose='ERROR')

# Crop epochs for the decoding pipeline from 0.5 to 2.5 after GO cue
epochs.crop(2.5, 4.5)

# Train the decoding pipeline
data = epochs.get_data()
labels = epochs.events[:, -1]

# Change labels to 0 and 1 - 0 class rest
labels[labels == 422] = 0
labels[labels == 421] = 1


############################# Data partitioning for training and testing #############################

train_data, test_data = [], []
train_labels, test_labels = [], []

for label in np.unique(labels):
    class_indices = np.where(labels == label)[0]
    train_size = int(len(class_indices) * 0.8)

    # Divide los índices sin mezclar
    train_indices = class_indices[:train_size]
    test_indices = class_indices[train_size:]

    # Añade los datos de entrenamiento y prueba
    train_data.append(data[train_indices])
    test_data.append(data[test_indices])
    train_labels.append(labels[train_indices])
    test_labels.append(labels[test_indices])

# Concatena las listas para obtener arrays finales
train_data = np.concatenate(train_data)
test_data = np.concatenate(test_data)
train_labels = np.concatenate(train_labels)
test_labels = np.concatenate(test_labels)

######################### TRAINING THE DECODING PIPELINE #########################

csp = CSP(n_components=6, reg='empirical', log=True, norm_trace=True, cov_est='epoch')
lda = LinearDiscriminantAnalysis()

train_data_csp = csp.fit_transform(train_data.astype(float), train_labels)
lda.fit(train_data_csp, train_labels)

print('Test accuracy: ', lda.score(csp.transform(test_data.astype(float)), test_labels))


######################### TRAIN WITH THE WHOLE DATA #########################

csp = CSP(n_components=6, reg='empirical', log=True, norm_trace=True, cov_est='epoch')
lda = LinearDiscriminantAnalysis()

data_csp = csp.fit_transform(data.astype(float), labels)
lda.fit(data_csp, labels)

print(f'Calibration: {lda.score(data_csp, labels)}')


# Save the decoding pipeline
trained_pipeline = {'csp': csp, 'lda': lda}

# Check the path to save the model
save_model_path = os.path.join(PATH_TO_SAVE_MODELS_EEG_MI, project_name, 'sub-' + training_subject_ID, 'ses-' + training_session_ID)
os.makedirs(save_model_path, exist_ok=True)

with open(os.path.join(save_model_path, 'CSP_LDA.pkl'), 'wb') as file:
    pickle.dump(trained_pipeline, file)