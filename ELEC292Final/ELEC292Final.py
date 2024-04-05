import pandas as pd
import numpy as np
import h5py
#used to clean up the axis of the ployts that have lots of time values
import pylab as pl
import matplotlib.pyplot as plt
from scipy.stats import mode, kurtosis, skew, t


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

# set csv into variables 

dj_walking = "dj_WalkingRawData.csv"
dj_jumping = "dj_JumpingRawData.csv"

isabel_walking = ["Isabel_Walking_RawData.csv", "Isabel_Walking_RawData2.csv"]
isabel_w_dataframe = [pd.read_csv(file) for file in isabel_walking]

isabel_jumping = ["Isabel_Jumping_RawData.csv", "Isabel_Jumping_RawData2.csv", "Isabel_Jumping_RawData3.csv", "Isabel_Jumping_RawData4.csv"]
isabel_j_dataframe = [pd.read_csv(file) for file in isabel_jumping]

lizzy_walking = "lizzy_WalkingRawData.csv"
lizzy_jumping = "lizzy_jumpingRawData.csv"


# reads the CSV files into data frames
dj_data_walking = pd.read_csv(dj_walking)
dj_data_jumping = pd.read_csv(dj_jumping)

isabel_data_walking = pd.concat(isabel_w_dataframe, axis=0, ignore_index=True)
isabel_data_jumping = pd.concat(isabel_j_dataframe, axis=0, ignore_index=True)

lizzy_data_walking = pd.read_csv(lizzy_walking)
lizzy_data_jumping = pd.read_csv(lizzy_jumping)

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


#WALKING X PLOTS
#dj data walking x plot
plt.figure(figsize = (10, 8))
#by default plot draws lines between the pointss/connects the data
dj_data_walking.plot(x='Time (s)', y='Linear Acceleration x (m/s^2)')
plt.title('DJ Acceleration in x')
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

#lizzy data walking x plot
plt.figure(figsize = (10, 8))
#by default plot draws lines between the pointss/connects the data
lizzy_data_walking.plot(x='Time (s)', y='Linear Acceleration x (m/s^2)')
plt.title('Lizzy Acceleration in x')
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

#isabel data walking x plot
plt.figure(figsize = (10, 8))
#by default plot draws lines between the pointss/connects the data
isabel_data_walking.plot(x='Time (s)', y='Linear Acceleration x (m/s^2)')
plt.title('Isabel Acceleration in x')
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

#WALKING Y PLOTS
#dj data walking y plot
plt.figure(figsize = (10, 8))
#by default plot draws lines between the pointss/connects the data
dj_data_walking.plot(x='Time (s)', y='Linear Acceleration y (m/s^2)')
plt.title('DJ Acceleration in y')
#adding labels to the axis
plt.xlabel("Time (s)")
plt.ylabel("Linear Acceleration y (m/s^2)")
#rotatess the labels 90 degrees to make it look better/no overlap
pl.xticks(rotation = 90)
#adding a legend
plt.legend(["Y acceleration"])
#formts the plot (fixes distance between elements and labels)
plt.tight_layout()
#function to show the plot
plt.show()

#lizzy data walking y plot
plt.figure(figsize = (10, 8))
#by default plot draws lines between the pointss/connects the data
lizzy_data_walking.plot(x='Time (s)', y='Linear Acceleration y (m/s^2)')
plt.title('Lizzy Acceleration in y')
#adding labels to the axis
plt.xlabel("Time (s)")
plt.ylabel("Linear Acceleration y (m/s^2)")
#rotatess the labels 90 degrees to make it look better/no overlap
pl.xticks(rotation = 90)
#adding a legend
plt.legend(["Y acceleration"])
#formts the plot (fixes distance between elements and labels)
plt.tight_layout()
#function to show the plot
plt.show()

#isabel data walking y plot
plt.figure(figsize = (10, 8))
#by default plot draws lines between the pointss/connects the data
isabel_data_walking.plot(x='Time (s)', y='Linear Acceleration y (m/s^2)')
plt.title('Isabel Acceleration in y')
#adding labels to the axis
plt.xlabel("Time (s)")
plt.ylabel("Linear Acceleration y (m/s^2)")
#rotatess the labels 90 degrees to make it look better/no overlap
pl.xticks(rotation = 90)
#adding a legend
plt.legend(["Y acceleration"])
#formts the plot (fixes distance between elements and labels)
plt.tight_layout()
#function to show the plot
plt.show()

#WALKING Z PLOTS
#dj data walking z plot
plt.figure(figsize = (10, 8))
#by default plot draws lines between the pointss/connects the data
dj_data_walking.plot(x='Time (s)', y='Linear Acceleration z (m/s^2)')
plt.title('DJ Acceleration in z')
#adding labels to the axis
plt.xlabel("Time (s)")
plt.ylabel("Linear Acceleration z (m/s^2)")
#rotatess the labels 90 degrees to make it look better/no overlap
pl.xticks(rotation = 90)
#adding a legend
plt.legend(["Z acceleration"])
#formts the plot (fixes distance between elements and labels)
plt.tight_layout()
#function to show the plot
plt.show()

#lizzy data walking z plot
plt.figure(figsize = (10, 8))
#by default plot draws lines between the pointss/connects the data
lizzy_data_walking.plot(x='Time (s)', y='Linear Acceleration z (m/s^2)')
plt.title('Lizzy Acceleration in z')
#adding labels to the axis
plt.xlabel("Time (s)")
plt.ylabel("Linear Acceleration z (m/s^2)")
#rotatess the labels 90 degrees to make it look better/no overlap
pl.xticks(rotation = 90)
#adding a legend
plt.legend(["Z acceleration"])
#formts the plot (fixes distance between elements and labels)
plt.tight_layout()
#function to show the plot
plt.show()

#isabel data walking z plot
plt.figure(figsize = (10, 8))
#by default plot draws lines between the pointss/connects the data
isabel_data_walking.plot(x='Time (s)', y='Linear Acceleration y (m/s^2)')
plt.title('Isabel Acceleration in z')
#adding labels to the axis
plt.xlabel("Time (s)")
plt.ylabel("Linear Acceleration z (m/s^2)")
#rotatess the labels 90 degrees to make it look better/no overlap
pl.xticks(rotation = 90)
#adding a legend
plt.legend(["Z acceleration"])
#formts the plot (fixes distance between elements and labels)
plt.tight_layout()
#function to show the plot
plt.show()

#JUMPING PLOTS
#JUMPING X PLOTS
#dj data jumping x plot
plt.figure(figsize = (10, 8))
#by default plot draws lines between the pointss/connects the data
dj_data_jumping.plot(x='Time (s)', y='Linear Acceleration x (m/s^2)')
plt.title('DJ jumping Acceleration in x')
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

#lizzy data jumping x plot
plt.figure(figsize = (10, 8))
#by default plot draws lines between the pointss/connects the data
lizzy_data_jumping.plot(x='Time (s)', y='Linear Acceleration x (m/s^2)')
plt.title('Lizzy jumping Acceleration in x')
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

#isabel data jumping x plot
plt.figure(figsize = (10, 8))
#by default plot draws lines between the pointss/connects the data
isabel_data_jumping.plot(x='Time (s)', y='Linear Acceleration x (m/s^2)')
plt.title('Isabel jumping Acceleration in x')
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

#JUMPING Y PLOTS
#dj data jumping y plot
plt.figure(figsize = (10, 8))
#by default plot draws lines between the pointss/connects the data
dj_data_jumping.plot(x='Time (s)', y='Linear Acceleration y (m/s^2)')
plt.title('DJ jumping Acceleration in y')
#adding labels to the axis
plt.xlabel("Time (s)")
plt.ylabel("Linear Acceleration y (m/s^2)")
#rotatess the labels 90 degrees to make it look better/no overlap
pl.xticks(rotation = 90)
#adding a legend
plt.legend(["Y acceleration"])
#formts the plot (fixes distance between elements and labels)
plt.tight_layout()
#function to show the plot
plt.show()

#lizzy data jumping y plot
plt.figure(figsize = (10, 8))
#by default plot draws lines between the pointss/connects the data
lizzy_data_jumping.plot(x='Time (s)', y='Linear Acceleration y (m/s^2)')
plt.title('Lizzy jumping Acceleration in y')
#adding labels to the axis
plt.xlabel("Time (s)")
plt.ylabel("Linear Acceleration y (m/s^2)")
#rotatess the labels 90 degrees to make it look better/no overlap
pl.xticks(rotation = 90)
#adding a legend
plt.legend(["Y acceleration"])
#formts the plot (fixes distance between elements and labels)
plt.tight_layout()
#function to show the plot
plt.show()

#isabel data jumping y plot
plt.figure(figsize = (10, 8))
#by default plot draws lines between the pointss/connects the data
isabel_data_jumping.plot(x='Time (s)', y='Linear Acceleration y (m/s^2)')
plt.title('Isabel Acceleration in y')
#adding labels to the axis
plt.xlabel("Time (s)")
plt.ylabel("Linear Acceleration y (m/s^2)")
#rotatess the labels 90 degrees to make it look better/no overlap
pl.xticks(rotation = 90)
#adding a legend
plt.legend(["Y acceleration"])
#formts the plot (fixes distance between elements and labels)
plt.tight_layout()
#function to show the plot
plt.show()

#JUMPING Z PLOTS
#dj data jumping z plot
plt.figure(figsize = (10, 8))
#by default plot draws lines between the pointss/connects the data
dj_data_jumping.plot(x='Time (s)', y='Linear Acceleration z (m/s^2)')
plt.title('DJ jumping Acceleration in z')
#adding labels to the axis
plt.xlabel("Time (s)")
plt.ylabel("Linear Acceleration z (m/s^2)")
#rotatess the labels 90 degrees to make it look better/no overlap
pl.xticks(rotation = 90)
#adding a legend
plt.legend(["Z acceleration"])
#formts the plot (fixes distance between elements and labels)
plt.tight_layout()
#function to show the plot
plt.show()

#lizzy data jumping z plot
plt.figure(figsize = (10, 8))
#by default plot draws lines between the pointss/connects the data
lizzy_data_jumping.plot(x='Time (s)', y='Linear Acceleration z (m/s^2)')
plt.title('Lizzy jumping Acceleration in z')
#adding labels to the axis
plt.xlabel("Time (s)")
plt.ylabel("Linear Acceleration z (m/s^2)")
#rotatess the labels 90 degrees to make it look better/no overlap
pl.xticks(rotation = 90)
#adding a legend
plt.legend(["Z acceleration"])
#formts the plot (fixes distance between elements and labels)
plt.tight_layout()
#function to show the plot
plt.show()

#isabel data jumping z plot
plt.figure(figsize = (10, 8))
#by default plot draws lines between the pointss/connects the data
isabel_data_walking.plot(x='Time (s)', y='Linear Acceleration y (m/s^2)')
plt.title('Isabel jumping Acceleration in z')
#adding labels to the axis
plt.xlabel("Time (s)")
plt.ylabel("Linear Acceleration z (m/s^2)")
#rotatess the labels 90 degrees to make it look better/no overlap
pl.xticks(rotation = 90)
#adding a legend
plt.legend(["Z acceleration"])
#formts the plot (fixes distance between elements and labels)
plt.tight_layout()
#function to show the plot
plt.show()

#FEATURE EXTRACTION + MORE VISUALIZATION

def extract_features(segment):
        features = {}
        features['maximum'] = np.max(segment)
        features['minimum'] = np.min(segment)
        features['range']= features['maximum'] - features['minimum']
        features['mean'] = np.mean(segment)
        features['median'] = np.median(segment)
        features['mode'] = mode(segment)[0][0]
        features['variance'] = np.var(segment)
        features['skewness'] = skew(segment)
        features['kurtosis'] = kurtosis(segment)
        features['sumOfInterval'] = np.sum(segment)

        confidence_level = 0.95
        n = len(segment)
        degreeOfFreedom = n - 1
        t_critical = t.ppf((1+confidence_level) / 2, degreeOfFreedom)
        standard_error = np.std(segment)/ np.sqrt(n)
        margin_of_error = t_critical * standard_error
        features['confidence_interval'] = (features['mean'] - margin_of_error, features['mean'] + margin_of_error)


        return features

def extract_features_from_dataset(dataset, window_size):
    features_list = []
    for start in range(0, len(dataset), window_size):
        end = start + window_size
        segment = dataset[start:end]
        if len(segment) == window_size:
            features = extract_features(segment)
            features_list.append(features)
    return features_list


# Example of how it can be used
'''
# # Example usage:
# Assuming 'data' is your signal data in a pandas DataFrame
# and 'window_size' is the size of the time window (5 seconds)
data = pd.read_csv("your_data.csv")
window_size = 100  # Assuming 100 data points represent 5 seconds

# Extracting features from the entire dataset
features_list = extract_features_from_dataset(data['Linear Acceleration x (m/s^2)'], window_size)

# Printing extracted features for the first segment as an example
# print(features_list[0])

'''

#extracting features
data = dj_data_walking
#statistical features !! need 3 more feautres
#creating an empty dataframe
features = pd.DataFrame(columns=['mean', 'std', 'max', 'kurtosis', 'skew', 'median', 'range'])
window_size = 125
features['mean'] = data.iloc[74829, -1].rolling(window=window_size).mean()
features['std'] = data.iloc[74829, -1].rolling(window=window_size).mean()
features['max'] = data.iloc[74829, -1].rolling(window=window_size).mean()
features['kurtosis'] = data.iloc[74829, -1].rolling(window=window_size).mean()
features['skew'] = data.iloc[74829, -1].rolling(window=window_size).mean()
features['median'] = data.iloc[74829, -1].rolling(window=window_size).mean()
features['range'] = data.iloc[74829, -1].rolling(window=window_size).mean()

#Plotting the features extracted
#Max X accerlation for dj



# Graphs Data like from lab 5
window_size = [5]
signalWithNoise = pd.read_csv() # data without filter

plt.plot(signalWithNoise, label = 'Orignial Data')

for window_size in window_size:
        filtered_signal = signalWithNoise.rolling(window = window_size).mean()
        plt.plot(filtered_signal, label=f'Moving Average {window_size}')

plt.legend()
plt.show()