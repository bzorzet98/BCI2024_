import mne
from brainflow.board_shim import BoardShim, BrainFlowInputParams

def setup_and_prepare_board(board_config):
    """
    Set up the board and prepare it for the experiment.
    """
    board_id = board_config['board_ID']
    port = board_config['port']
    ch_list = board_config['ch_list']

    # BoardShim.enable_dev_board_logger()  #  To show the logs in the console
    params = BrainFlowInputParams()
    params.serial_port = port
    board = BoardShim(board_id, params)
    exg_channels = BoardShim.get_exg_channels(board_id)
    sampling_rate = BoardShim.get_sampling_rate(board_id)
    info = mne.create_info(ch_names=ch_list, sfreq=sampling_rate, ch_types='eeg')
    board.prepare_session()

    return board, info, exg_channels