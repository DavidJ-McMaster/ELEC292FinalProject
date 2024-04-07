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

from Part_1 import *
from Part_2 import *

def noise_filtering(file, window_size):
    #window = [5]
    noise_filter = pd.read_csv(file)
    plt.plot(noise_filter, label='Original Data')
    filtered_signal = noise_filter.rolling(window=window_size).mean()
    plt.plot(filtered_signal, label=f'Moving Average {window_size}')

    plt.legend()
    plt.show()


# feature extraction and more visualization?
def extract_features(segment):
    features = pd.DataFrame(
        columns=['maximum', 'minimum', 'range', 'mean', 'median', 'mode', 'variance', 'skewness', 'kurtosis',
                 'sumOfInterval', 'confidence_interval'])
    features['maximum'] = np.max(segment)
    features['minimum'] = np.min(segment)
    features['range'] = features['maximum'] - features['minimum']
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
    t_critical = t.ppf((1 + confidence_level) / 2, degreeOfFreedom)
    standard_error = np.std(segment) / np.sqrt(n)
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


# calling feature extraction for each set of data
window_size = 5

# DJ features
dj_walking_features = extract_features_from_dataset(dj_data_walking, window_size)
dj_jumping_features = extract_features_from_dataset(dj_data_jumping, window_size)
# Lizzy features
lizzy_walking_features = extract_features_from_dataset(lizzy_data_walking, window_size)
lizzy_jumping_features = extract_features_from_dataset(lizzy_data_jumping, window_size)
# Isabel features
isabel_walking_features = extract_features_from_dataset(isabel_data_walking, window_size)
isabel_jumping_features = extract_features_from_dataset(isabel_data_jumping, window_size)

# turning the extracted features into dataframes
dj_walking_df = pd.concat(dj_walking_features)
dj_jumping_df = pd.concat(dj_jumping_features)

lizzy_walking_df = pd.concat(lizzy_walking_features)
lizzy_jumping_df = pd.concat(lizzy_jumping_features)

isabel_walking_df = pd.concat(isabel_walking_features)
isabel_jumping_df = pd.concat(isabel_jumping_features)

# Plotting the mean acceleration for walking vs jumping for each person
plt.figure(15, 8)
plt.subplot(3, 1, 1)
plt.xlabel("Time (s)")
plt.ylabel("mean acceleration DJ")
plt.plot(x='Time (s)', y=dj_walking_df['mean'], label="walking")
plt.plot(x='Time (s)', y=dj_jumping_df['mean'], label="jumping")

plt.figure(15, 8)
plt.subplot(3, 1, 2)
plt.xlabel("Time (s)")
plt.ylabel("mean acceleration Lizzy")
plt.plot(x='Time (s)', y=lizzy_walking_df['mean'], label="walking")
plt.plot(x='Time (s)', y=lizzy_jumping_df['mean'], label="jumping")

plt.figure(15, 8)
plt.subplot(3, 1, 3)
plt.xlabel("Time (s)")
plt.ylabel("mean acceleration Isabel")
plt.plot(x='Time (s)', y=isabel_walking_df['mean'], label="walking")
plt.plot(x='Time (s)', y=isabel_jumping_df['mean'], label="jumping")

plt.tight_layout()
plt.show()

# creating histograms for each feature of jumping vs walking with points for each 5 second interval
# combining the walking and jumping data into one dataframe for each person

# DJ data comparison walking
data = dj_walking_df.iloc[:, 0:9]
labels = dj_walking_df.iloc[:, 9]
print(labels)
fig, ax = plt.subplots(ncols=3, nrows=3, figsize=(25, 15))
data.hist(ax=ax.flatten()[0:9])
fig.tight_layout()
plt.show()

# DJ data comparison jumping
data = dj_jumping_df.iloc[:, 0:9]
labels = dj_jumping_df.iloc[:, 9]
print(labels)
fig, ax = plt.subplots(ncols=3, nrows=3, figsize=(25, 15))
data.hist(ax=ax.flatten()[0:9])
fig.tight_layout()
plt.show()

# Lizzy data comparison walking
data = lizzy_walking_df.iloc[:, 0:9]
labels = lizzy_walking_df.iloc[:, 9]
print(labels)
fig, ax = plt.subplots(ncols=3, nrows=3, figsize=(25, 15))
data.hist(ax=ax.flatten()[0:9])
fig.tight_layout()
plt.show()

# Lizzy data comparison jumping
data = lizzy_jumping_df.iloc[:, 0:9]
labels = lizzy_jumping_df.iloc[:, 9]
print(labels)
fig, ax = plt.subplots(ncols=3, nrows=3, figsize=(25, 15))
data.hist(ax=ax.flatten()[0:9])
fig.tight_layout()
plt.show()

# Isabel data comparison walking
data = isabel_walking_df.iloc[:, 0:9]
labels = isabel_walking_df.iloc[:, 9]
print(labels)
fig, ax = plt.subplots(ncols=3, nrows=3, figsize=(25, 15))
data.hist(ax=ax.flatten()[0:9])
fig.tight_layout()
plt.show()

# Isabel data comparison jumping
data = isabel_jumping_df.iloc[:, 0:9]
labels = isabel_jumping_df.iloc[:, 9]
print(labels)
fig, ax = plt.subplots(ncols=3, nrows=3, figsize=(25, 15))
data.hist(ax=ax.flatten()[0:9])
fig.tight_layout()
plt.show()