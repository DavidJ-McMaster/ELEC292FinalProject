import pandas as pd
import numpy as np
import h5py

# reads the CSV files into data frames
dj_data_walking = pd.read_csv("")
dj_data_jumping = pd.read_csv("")
isabel_data_walking = pd.read_csv("")
isabel_data_jumping = pd.read_csv("")
lizzy_data_walking = pd.read_csv("")
lizzy_data_jumping = pd.read_csv("")

# merged data
dj_data_combined = pd.concat([dj_data_jumping, dj_data_walking], axis=0)
isabel_data_combined = pd.concat([isabel_data_jumping, isabel_data_walking], axis=0)
lizzy_data_combined = pd.concat([lizzy_data_jumping, lizzy_data_walking], axis=0)

with h5py.File("data.h5", 'w') as hdf_file:

    Member1 = hdf_file.create_group("Member1")
    Member2 = hdf_file.create_group("Member2")
    Member3 = hdf_file.create_group("Member3")

    Member1.create_dataset("data", data=dj_data_combined)
    Member2.create_dataset("data", data=isabel_data_combined)
    Member3.create_dataset("data", data=lizzy_data_combined)

    train_ratio = 0.9
    test_ratio = 0.1
