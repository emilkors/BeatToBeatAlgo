import os
import h5py
def create_h5_if_not_exists(file_path):
    # Pre-create the file if it doesn't already exist
    if not os.path.exists(file_path):
        with h5py.File(file_path, 'w'):
            # The file is empty at this point, and we can add groups or datasets as needed
            print(f"Empty HDF5 file '{file_path}' created successfully.")
    else:
        raise FileExistsError(f"The file '{file_path}' already exists. Cannot overwrite.") 