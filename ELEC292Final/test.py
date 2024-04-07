import pandas as pd
import numpy as np
import h5py
import matplotlib.pyplot as plt

# Function to segment data into 5-second windows
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

# Read data from CSV files into DataFrames
dj_data_walking = pd.read_csv("dj_WalkingRawData.csv")
isabel_data_walking = pd.read_csv("Isabel_Walking_RawData2.csv")
lizzy_data_walking = pd.read_csv("lizzy_NewWalkingData.csv")

# Segment data into 5-second windows
dj_windows = segment_into_windows(dj_data_walking)
isabel_windows = segment_into_windows(isabel_data_walking)
lizzy_windows = segment_into_windows(lizzy_data_walking)

# Combine windows from all members
combined_windows = dj_windows + isabel_windows + lizzy_windows

# Shuffle the combined windows
np.random.shuffle(combined_windows)

# Split data into training and testing sets (90% for training, 10% for testing)
train_size = int(0.9 * len(combined_windows))
train_windows = combined_windows[:train_size]
test_windows = combined_windows[train_size:]

# Store the datasets in HDF5 file
with h5py.File("data.h5", 'w') as hdf_file:
    train_group = hdf_file.create_group("train")
    test_group = hdf_file.create_group("test")
    
    # Store training data
    for i, window_data in enumerate(train_windows):
        train_group.create_dataset(f"window_{i}", data=window_data)
    
    # Store testing data
    for i, window_data in enumerate(test_windows):
        test_group.create_dataset(f"window_{i}", data=window_data)

# Visualize the segmented data
plt.figure(figsize=(10, 6))

# Plot train data
train_data = pd.concat(train_windows)
plt.scatter(train_data['Time (s)'], train_data['Linear Acceleration x (m/s^2)'], label='Train', alpha=0.5)

# Plot test data
test_data = pd.concat(test_windows)
plt.scatter(test_data['Time (s)'], test_data['Linear Acceleration x (m/s^2)'], label='Test', alpha=0.5)

plt.xlabel('Time (s)')
plt.ylabel('Linear Acceleration x (m/s^2)')
plt.title('Segmented Data Scatter Plot')
plt.legend()
plt.grid(True)
plt.show()
