import numpy as np
import os
import json
import mne
import pickle
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from mne.decoding import CSP
import pandas as pd

mne.set_log_level(verbose='warning') #to avoid info at terminal

root_path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))

script_path = os.path.dirname(os.path.realpath(__file__))
os.chdir(script_path)
root_path = os.path.dirname(script_path)

# Load the training config
with open(os.path.join(root_path, 'configs', 'training_config.json')) as file:
    training_config = json.load(file)

# Load the training data
training_subject_ID = training_config['subject_ID']
training_session_ID = training_config['session_ID']
training_runs_ID = training_config['runs_ID']
training_task = training_config['task']
project_name = training_config['project_name']


{
	"subject_ID": "001",
	"session_ID": "0",
	"runs_ID": ["1"],
	"task": "calibration",
    "project_name": "MIBCIproject"
}




epochs = []
for run in training_runs_ID:
    # Load the training data
    training_folder_path = os.path.join(root_path, 'data', project_name, 'sub-' + training_subject_ID, 'ses-' + training_session_ID, 'eeg')
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

# Fit CSP
csp = CSP(n_components=6, reg='empirical', log=True, norm_trace=True, cov_est='epoch')
Gtr = csp.fit_transform(data.astype(float), labels)
# Fit LDA
lda = LinearDiscriminantAnalysis()
lda.fit(Gtr, labels)
print('Calibration accuracy: ', lda.score(Gtr, labels))

# Save the decoding pipeline
trained_pipeline = {'csp': csp, 'lda': lda}
# Check the path to save the model
save_model_path = os.path.join(root_path, 'models', project_name, 'sub-' + training_subject_ID, 'ses-' + training_session_ID)
if not os.path.exists(save_model_path):
    os.makedirs(save_model_path)
with open(os.path.join(save_model_path, 'trained_pipeline.pkl'), 'wb') as file:
    pickle.dump(trained_pipeline, file)