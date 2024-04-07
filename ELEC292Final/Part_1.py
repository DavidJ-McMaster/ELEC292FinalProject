import pandas as pd
import numpy as np
import h5py
import pylab as pl
import matplotlib.pyplot as plt
from scipy.stats import mode, kurtosis, skew, t
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import random

# this file reads the data, segments it and splits it testing/training

def segment_into_windows(df, window_size=5):
    windows = []
    start_time = df['Time (s)'].min()  # Start time of the first window
    end_time = start_time + window_size

    while end_time <= df['Time (s)'].max():
        window_data = df[(df['Time (s)'] >= start_time) & (df['Time (s)'] < end_time)]
        windows.append(window_data)
        start_time = end_time
        end_time += window_size

    return windows

def calculate_average_point(windows):
    average_points = []
    for window_data in windows:
        average_point = window_data.mean()  # Calculate mean for each feature
        average_points.append(average_point)
    return average_points


dj_walking = "dj_WalkingRawData.csv"
dj_jumping = "dj_JumpingRawData.csv"

isabel_walking = "Isabel_Walking_RawData2.csv"
isabel_jumping = "Isabel_Jumping_RawData4.csv"

lizzy_walking = "lizzy_NewWalkingData.csv"
lizzy_jumping = "lizzy_NewJumping.csv"

# segment into 5 second intervals

dj_data_walking = pd.read_csv(dj_walking)
dj_data_jumping = pd.read_csv(dj_jumping)
isabel_data_walking = pd.read_csv(isabel_walking)
isabel_data_jumping = pd.read_csv(isabel_jumping)
lizzy_data_walking = pd.read_csv(lizzy_walking)
lizzy_data_jumping = pd.read_csv(lizzy_jumping)

dj_walking_segments = segment_into_windows(dj_data_walking)
isabel_walking_segments = segment_into_windows(isabel_data_walking)
lizzy_walking_segments = segment_into_windows(lizzy_data_walking)

dj_jumping_segments = segment_into_windows(dj_data_jumping)
isabel_jumping_segments = segment_into_windows(isabel_data_jumping)
lizzy_jumping_segments = segment_into_windows(lizzy_data_jumping)

dj_walking_data = pd.concat(dj_walking_segments)
lizzy_walking_data = pd.concat(lizzy_walking_segments)
isabel_walking_data = pd.concat(isabel_walking_segments)

dj_jumping_data = pd.concat(dj_jumping_segments)
lizzy_jumping_data = pd.concat(lizzy_jumping_segments)
isabel_jumping_data = pd.concat(isabel_jumping_segments)

# Sort the dataframes by 'Time (s)'
dj_walking_sorted = dj_walking_data.sort_values(by='Time (s)')
lizzy_walking_sorted = lizzy_walking_data.sort_values(by='Time (s)')
isabel_walking_sorted = isabel_walking_data.sort_values(by='Time (s)')

dj_jumping_sorted = dj_jumping_data.sort_values(by='Time (s)')
lizzy_jumping_sorted = lizzy_jumping_data.sort_values(by='Time (s)')
isabel_jumping_sorted = isabel_jumping_data.sort_values(by='Time (s)')


combined_walking = dj_walking_segments + isabel_walking_segments + lizzy_walking_segments
combined_walking_average = calculate_average_point(combined_walking)

combined_jumping = dj_jumping_segments + isabel_jumping_segments + lizzy_jumping_segments
combined_jumping_average = calculate_average_point(combined_jumping)

np.random.shuffle(combined_jumping_average)
np.random.shuffle(combined_walking_average)
np.random.shuffle(combined_walking)
np.random.shuffle(combined_jumping)


walking_training, walking_testing = train_test_split(combined_walking_average, test_size=0.1, shuffle=True, random_state=5)
jumping_training, jumping_testing = train_test_split(combined_jumping_average, test_size=0.1, shuffle=True, random_state=5)

# Store the datasets in HDF5 file
with h5py.File("data.h5", 'w') as hdf_file:
    train_group = hdf_file.create_group("train")
    test_group = hdf_file.create_group("test")

    # Store training data
    for i, window_data in enumerate(walking_training):
        train_group.create_dataset(f"window_{i}", data=window_data)

    # Store testing data
    for i, window_data in enumerate(walking_testing):
        test_group.create_dataset(f"window_{i}", data=window_data)





