import pickle

def load_calibration_data(file_path):
    with open(file_path, 'rb') as f:
        return pickle.load(f)