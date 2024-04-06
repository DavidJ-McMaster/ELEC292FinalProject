import pandas as pd
import numpy as np
import h5py
#used to clean up the axis of the ployts that have lots of time values
import pylab as pl
import matplotlib.pyplot as plt
from scipy.stats import mode, kurtosis, skew, t
from sklearn import preprocessing


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

isabel_jumping = ["Isabel_Jumping_Raw Data.csv", "Isabel_Jumping_RawData2.csv", "Isabel_Jumping_RawData3.csv", "Isabel_Jumping_RawData4.csv"]
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


with h5py.File("data.h5", 'w') as hdf_file:

    Member1 = hdf_file.create_group("Member1") # dj
    Member2 = hdf_file.create_group("Member2") # isabel
    Member3 = hdf_file.create_group("Member3") # lizzy

    Member1.create_dataset("walking", data=dj_data_walking)
    Member1.create_dataset("jumping", data=dj_data_jumping)

    Member2.create_dataset("walking", data=isabel_data_walking)
    Member2.create_dataset("jumping", data=isabel_data_jumping)

    Member3.create_dataset("walking", data=lizzy_data_walking)
    Member3.create_dataset("jumping", data=lizzy_data_jumping)

    hdf_file.attrs['train_ratio'] = 0.9
    hdf_file.attrs['test_ratio'] = 0.1

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
        features = pd.DataFrame(columns=['maximum', 'minimum', 'range', 'mean', 'median', 'mode', 'variance', 'skewness','kurtosis','sumOfInterval','confidence_interval'])
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


#calling feature extraction for each set of data
window_size = 5

#DJ features
dj_walking_features = extract_features_from_dataset(dj_data_walking, window_size)
dj_jumping_features = extract_features_from_dataset(dj_data_jumping, window_size)
#Lizzy features
lizzy_walking_features = extract_features_from_dataset(lizzy_data_walking, window_size)
lizzy_jumping_features = extract_features_from_dataset(lizzy_data_jumping, window_size)
#Isabel features
isabel_walking_features = extract_features_from_dataset(isabel_data_walking, window_size)
isabel_jumping_features = extract_features_from_dataset(isabel_data_jumping, window_size)

#turning the extracted features into dataframes
dj_walking_df = pd.concat(dj_walking_features)
dj_jumping_df = pd.concat(dj_jumping_features)

lizzy_walking_df = pd.concat(lizzy_walking_features)
lizzy_jumping_df = pd.concat(lizzy_jumping_features)

isabel_walking_df = pd.concat(isabel_walking_features)
isabel_jumping_df = pd.concat(isabel_jumping_features)

#Plotting the mean acceleration for walking vs jumping for each person
plt.figure(15, 8)
plt.subplot(3, 1, 1)
plt.xlabel("Time (s)")
plt.ylabel("mean acceleration DJ")
plt.plot(x='Time (s)', y= dj_walking_df['mean'], label = "walking")
plt.plot(x= 'Time (s)', y=dj_jumping_df['mean'], label = "jumping")

plt.figure(15, 8)
plt.subplot(3, 1, 2)
plt.xlabel("Time (s)")
plt.ylabel("mean acceleration Lizzy")
plt.plot(x='Time (s)', y=lizzy_walking_df['mean'], label = "walking")
plt.plot(x='Time (s)', y=lizzy_jumping_df['mean'], label = "jumping")

plt.figure(15, 8)
plt.subplot(3, 1, 3)
plt.xlabel("Time (s)")
plt.ylabel("mean acceleration Isabel")
plt.plot(x='Time (s)', y=isabel_walking_df['mean'], label = "walking")
plt.plot(x='Time (s)', y=isabel_jumping_df['mean'], label = "jumping")

plt.tight_layout()
plt.show()

#creating histograms for each feature of jumping vs walking with points for each 5 second interval
#combining the walking and jumping data into one dataframe for each person

#DJ data comparison walking
data = dj_walking_df.iloc[:, 0:9]
labels = dj_walking_df.iloc[:, 9]
print(labels)
fig, ax = plt.subplots(ncols=3, nrows=3, figsize=(25, 15))
data.hist(ax=ax.flatten()[0:9])
fig.tight_layout()
plt.show()

#DJ data comparison jumping
data = dj_jumping_df.iloc[:, 0:9]
labels = dj_jumping_df.iloc[:, 9]
print(labels)
fig, ax = plt.subplots(ncols=3, nrows=3, figsize=(25, 15))
data.hist(ax=ax.flatten()[0:9])
fig.tight_layout()
plt.show()

#Lizzy data comparison walking
data = lizzy_walking_df.iloc[:, 0:9]
labels = lizzy_walking_df.iloc[:, 9]
print(labels)
fig, ax = plt.subplots(ncols=3, nrows=3, figsize=(25, 15))
data.hist(ax=ax.flatten()[0:9])
fig.tight_layout()
plt.show()

#Lizzy data comparison jumping
data = lizzy_jumping_df.iloc[:, 0:9]
labels = lizzy_jumping_df.iloc[:, 9]
print(labels)
fig, ax = plt.subplots(ncols=3, nrows=3, figsize=(25, 15))
data.hist(ax=ax.flatten()[0:9])
fig.tight_layout()
plt.show()

#Isabel data comparison walking
data = isabel_walking_df.iloc[:, 0:9]
labels = isabel_walking_df.iloc[:, 9]
print(labels)
fig, ax = plt.subplots(ncols=3, nrows=3, figsize=(25, 15))
data.hist(ax=ax.flatten()[0:9])
fig.tight_layout()
plt.show()

#Isabel data comparison jumping
data = isabel_jumping_df.iloc[:, 0:9]
labels = isabel_jumping_df.iloc[:, 9]
print(labels)
fig, ax = plt.subplots(ncols=3, nrows=3, figsize=(25, 15))
data.hist(ax=ax.flatten()[0:9])
fig.tight_layout()
plt.show()

""" 
plt.figure(figsize=(15, 10))


for i, featureA in enumerate(features_to_plot):
     for j, featureB in enumerate(features_to_plot):
          #creating the number of subplots and selecting the subplot to put the current data in
          plt.subplot(number_features, number_features, i)
          plt.scatter(dj_walking_features[featureA], dj_walking_features[featureB], label = 'Walking', colour='blue')
          plt.scatter(dj_jumping_features[featureA], dj_jumping_features[featureB], label = 'Jumping', colour='red')
          plt.xlabel(featureA)
          plt.ylabel(featureB)

plt.title("DJ features correlation")
plt.tight_layout()
plt.legend()
plt.show()

#lizzy features
lizzy_walking_features = extract_features_from_dataset(lizzy_data_walking, window_size)
lizzy_jumping_features = extract_features_from_dataset(lizzy_data_jumping, window_size)

#plotting correlation between extracted features walking vs jumping

plt.figure(figsize=(15, 10))

for i, featureA in enumerate(features_to_plot):
     for j, featureB in enumerate(features_to_plot):
          #creating the number of subplots and selecting the subplot to put the current data in
          plt.subplot(number_features, number_features, i)
          plt.scatter(lizzy_walking_features[featureA], lizzy_walking_features[featureB], label = 'Walking', colour='blue')
          plt.scatter(lizzy_jumping_features[featureA], lizzy_jumping_features[featureB], label = 'Jumping', colour='red')
          plt.xlabel(featureA)
          plt.ylabel(featureB)

plt.title("Lizzy features correlation")
plt.tight_layout()
plt.legend()
plt.show()

#Isabel features
isabel_walking_features = extract_features_from_dataset(isabel_data_walking, window_size)
isabel_jumping_features = extract_features_from_dataset(isabel_data_jumping, window_size)

#plotting correlation between extracted features walking vs jumping

plt.figure(figsize=(15, 10))

for i, featureA in enumerate(features_to_plot):
     for j, featureB in enumerate(features_to_plot):
          #creating the number of subplots and selecting the subplot to put the current data in
          plt.subplot(number_features, number_features, i)
          plt.scatter(isabel_walking_features[featureA], isabel_walking_features[featureB], label = 'Walking', colour='blue')
          plt.scatter(isabel_jumping_features[featureA], isabel_jumping_features[featureB], label = 'Jumping', colour='red')
          plt.xlabel(featureA)
          plt.ylabel(featureB)

plt.title("Isabel features correlation")
plt.tight_layout()
plt.legend()
plt.show() """

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

""" #extracting features
data = dj_data_walking
#statistical features !! need 3 more feautres
#creating an empty dataframe
features = pd.DataFrame(columns=['mean', 'std', 'max', 'kurtosis', 'skew', 'median', 'range'])
window_size = 125
features['mean'] = data.iloc[74829, -1].rolling(window=window_size).mean()
features['std'] = data.iloc[74829, -1].rolling(window=window_size).m()
features['max'] = data.iloc[74829, -1].rolling(window=window_size).mean()
features['kurtosis'] = data.iloc[74829, -1].rolling(window=window_size).mean()
features['skew'] = data.iloc[74829, -1].rolling(window=window_size).mean()
features['median'] = data.iloc[74829, -1].rolling(window=window_size).mean()
features['range'] = data.iloc[74829, -1].rolling(window=window_size).mean()
 """



# Graphs Data like from lab 5
window_size = [5]
signalWithNoise = pd.read_csv() # data without filter

plt.plot(signalWithNoise, label = 'Orignial Data')

for window_size in window_size:
        filtered_signal = signalWithNoise.rolling(window = window_size).mean()
        plt.plot(filtered_signal, label=f'Moving Average {window_size}')

plt.legend()
plt.show()