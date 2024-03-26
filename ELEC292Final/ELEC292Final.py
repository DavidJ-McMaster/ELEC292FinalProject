import pandas as pd
import numpy as np
import h5py
#used to clean up the axis of the ployts that have lots of time values
import pylab as pl
import matplotlib.pyplot as plt


# splitting the data into segmented 5-second windows
def segmentation(signal,
                 window_size=5,
                 overlap=0):
    segments = [] # empty list to hold segmented data
    for start in range(0, len(signal), window_size-overlap):
        end = start + window_size
        segment = signal[start:end]
        if len(segment) == window_size:
            segments.append(segment) # adds segment to the list once it is as long as the 5-second interval
    return segments


# reads the CSV files into data frames
dj_data_walking = pd.read_csv("")
dj_data_jumping = pd.read_csv("")
isabel_data_walking = pd.read_csv("")
isabel_data_jumping = pd.read_csv("")
lizzy_data_walking = pd.read_csv("")
lizzy_data_jumping = pd.read_csv("")

"""
dj_data_combined = pd.concat([dj_data_jumping, dj_data_walking], axis=0)
isabel_data_combined = pd.concat([isabel_data_jumping, isabel_data_walking], axis=0)
lizzy_data_combined = pd.concat([lizzy_data_jumping, lizzy_data_walking], axis=0)
"""

with h5py.File("data.h5", 'w') as hdf_file:

    Member1 = hdf_file.create_group("Member1")
    Member2 = hdf_file.create_group("Member2")
    Member3 = hdf_file.create_group("Member3")

    Member1.create_dataset("data", data=dj_data_combined)
    Member2.create_dataset("data", data=isabel_data_combined)
    Member3.create_dataset("data", data=lizzy_data_combined)

    train_ratio = 0.9
    test_ratio = 0.1

#data visualization

#plot x, y and z vs time for each data type
# 6 plots per person

#dj data walking x plot
plt.firgure(figszie = (10, 8))
#by default plot draws lines between the pointss/connects the data
dj_data_walking.plot(x='Time (s)', y='Linear Acceleration x (m/s^2)')
plt.title('Acceleration in x')
#adding labels to the axis
plt.xlabel("Time (s)")
plt.ylabel("Linear Acceleration x (m/s^2)")
#rotatess the labels 90 degrees to make it look better/no overlap
pl.xticks(rotation = 90)
#adding a legend
plt.legend(["X acceleration"])
#formts the plot (fixes distance between elements and labels)
plt.tight_layout()
#function to show the plot
plt.show()


# Graphs Data like from lab 5
window_size = [5]
signalWithNoise = pd.read_csv() # data without filter

plt.plot(signalWithNoise, label = 'Orignial Data')

for window_size in window_size:
        filtered_signal = signalWithNoise.rolling(window = window_size).mean()
        plt.plot(filtered_signal, label=f'Moving Average {window_size}')

plt.legend()
plt.show()