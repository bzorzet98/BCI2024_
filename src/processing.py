import time
import mne
from mne.decoding import CSP
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

def predict_one_trial(board, exg_channels_indices, mne_info, trained_pipeline):
    """
    Predicts the class of the motor imagery task for one trial.
    """
    time.sleep(2.5)

    # Get data from -2 s to 2.5 s
    sampling_rate = mne_info['sfreq']
    n_samples = int(4.5*sampling_rate)
    data = board.get_current_board_data(n_samples)
    data = data[exg_channels_indices]   # Keep only the EEG channels
    
    # Process trial
    data = data/1000000   # Convert from uV to V for MNE
    data = data.reshape(1, data.shape[0], data.shape[1])

    trial_epoch = mne.EpochsArray(data, mne_info)
    # Remove last channel
    trial_epoch.drop_channels(['NA'])
    trial_epoch.filter(8, 30)
    # Crop from 0.5 s to 2.5 s post go cue.
    # Here time is referenced to the trial indication marker
    trial_epoch.crop(2.5, 4.5)
    trial_array = trial_epoch.get_data()

    # CSP transformation
    csp = trained_pipeline['csp']
    Gte = csp.transform(trial_array)

    # LDA classification
    lda = trained_pipeline['lda']
    y_pred = lda.predict(Gte)

    return trial_array, y_pred
