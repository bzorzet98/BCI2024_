import time
import mne
import numpy as np
from sklearn.cross_decomposition import CCA

def predict_one_trial_MI(board, exg_channels_indices, mne_info, trained_pipeline):
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

def generate_reference_signals(target_freq, sampling_rate, n_samples, n_harmonics=6):
    # Generate sinusoidal reference templates for CCA for the given flicker frequency and number of harmonics
    reference_signals = []
    t = np.arange(0, (n_samples/(sampling_rate)), step=1.0/(sampling_rate))
    for i in range(1, n_harmonics+1):
        reference_signals.append(np.sin(2*np.pi*i*target_freq*t))
        reference_signals.append(np.cos(2*np.pi*i*target_freq*t))
    return reference_signals
            
def find_corr(n_components, eeg_data, freq):
    # Perform Canonical correlation analysis (CCA)
    # eeg_data - consists of the EEG
    # freq - set of sinusoidal reference templates corresponding to the flicker frequency
    cca = CCA(n_components)
    corr = np.zeros(n_components)
    result = np.zeros((freq.shape)[0])
    for freq_idx in range((freq.shape)[0]):
        # Fit the CCA model to the EEG data and reference signals
        cca.fit(np.squeeze(eeg_data.T), np.squeeze(freq[freq_idx]).T)
        # Transform the EEG data and reference signals into the canonical space
        O1_a, O1_b = cca.transform(np.squeeze(eeg_data.T), np.squeeze(freq[freq_idx]).T)
        for ind_val in range(n_components):
            corr[ind_val] = np.corrcoef(O1_a[:, ind_val], O1_b[:, ind_val])[0, 1]
        result[freq_idx] = np.max(corr)
    return result

def predict_one_trial_SSVEP(board, exg_channels_indices, mne_info):
    """
    Predicts the class of SSVEP for one trial.
    """
    window_length = 3  # seconds
    n_CCA_components = 1
    n_harmonics = 6
    frequencies = [8.5, 10, 12, 15] # left: 8.5 Hz, right: 10 Hz, up: 12Hz, down: 15Hz
    time.sleep(window_length + 2)

    sampling_rate = mne_info['sfreq']
    # Get data from 0 s to window_length + 1 s data
    n_samples = int((window_length+2)*sampling_rate)
    data = board.get_current_board_data(n_samples)
    data = data[exg_channels_indices]   # Keep only the EEG channels
    
    # Process trial
    data = data/1000000   # Convert from uV to V for MNE
    data = data.reshape(1, data.shape[0], data.shape[1])

    trial_epoch = mne.EpochsArray(data, mne_info)
    # Pick only O1 and O2 channels
    trial_epoch.pick(['Fp1', 'Fp2'])
    # Bandpass filter from 1 to 40 Hz
    trial_epoch.filter(1, 20)
    # Crop from 1 s to window_length + 1 s post go cue.
    # Here time is referenced to the trial indication marker
    trial_epoch.crop(1, window_length + 1)
    trial_array = trial_epoch.get_data()

    # Generate a vector of sinusoidal reference templates for all SSVEP flicker frequencies
    freq = np.array([generate_reference_signals(f, sampling_rate, trial_array.shape[2], n_harmonics) for f in frequencies])
    # Compute CCA 
    CCA_output = find_corr(n_CCA_components, trial_array, freq)
    # Find the maximum canonical correlation coefficient and corresponding class for the given SSVEP/EEG data
    y_pred = np.argmax(CCA_output)

    return trial_array, y_pred
