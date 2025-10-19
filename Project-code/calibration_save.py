# --- Save and Load Stereo Calibration Data Script ---
# This script saves the results of the stereo camera calibration
# and reload them later for use in other modules (like depth calculation).
# It uses Python's pickle module for binary serialisation.

import pickle

def save_calibration_data(calibration_data, filename='stereo_calibration.pkl'):
    """
    Save calibration data to a file using pickle.
    
    Args:
        calibration_data (dict): The dictionary containing calibration results.
        filename (str): The filename where the calibration data will be saved.
    """
    with open(filename, 'wb') as f:
        pickle.dump(calibration_data, f)
    print(f"Calibration data successfully saved to '{filename}'.")

def load_calibration_data(filename='stereo_calibration.pkl'):
    """
    Load calibration data from a pickle file.
    
    Args:
        filename (str): The filename from which to load the calibration data.
    
    Returns:
        dict: The loaded calibration data.
    """
    with open(filename, 'rb') as f:
        data = pickle.load(f)
    print(f"Calibration data successfully loaded from '{filename}'.")
    return data

if __name__ == '__main__':
    # Example usage: Perform calibration and save the result.
    from calibration_compute import calibrate_stereo_cameras

    print("Starting stereo calibration process...")
    calibration_data = calibrate_stereo_cameras()
    save_calibration_data(calibration_data)