import csv
import numpy as np
import os
import pandas as pd
from sklearn import datasets
from sklearn.model_selection import train_test_split

def read_dataset():
    df = pd.read_csv('https://media.githubusercontent.com/media/Benz-Poobua/ESS-469-Project/main/bulk_event_magnitude_phase_nwf_FV.csv', parse_dates=['datetime', 'arrdatetime'])
    df.index = df.arid
    latll, lonll = 42.0, -125.5
    latur, lonur = 49.0, -122.0
    df= df[
        (df.lat >= latll)
        & (df.lat <= latur)
        & (df.lon >= lonll)
        & (df.lon <= lonur)
    ]
    feature_df = df.copy().iloc[:, -140:]
    data_arr = np.empty((0, 140), dtype=np.float16)
    target_arr = np.empty((0, 1), dtype=np.float16)
    # dataframe of the last 140 columns to numpy array
    for i in df.arid.unique():
        data_arr = np.append(data_arr, np.array([feature_df.loc[i]], dtype=np.float16), axis=0)
        target_arr = np.append(target_arr, np.array([[df.loc[i].magnitude]], dtype=np.float16), axis=0)
    return data_arr, target_arr

def save_array(arr, filename):
    # save the array to a file
    np.save(filename, arr)

def import_array(filename):
    # import the array from the file
    arr = np.load(filename)
    return arr

# check if the file exists
if os.path.isfile('bulk_event_magnitude_phase_nwf_FV.npy') and os.path.isfile('bulk_event_magnitude_phase_nwf_FV_target.npy'):
    data_arr = import_array('bulk_event_magnitude_phase_nwf_FV.npy')
    target_arr = import_array('bulk_event_magnitude_phase_nwf_FV_target.npy')
else:
    data_arr, target_arr = read_dataset()
    save_array(data_arr, 'bulk_event_magnitude_phase_nwf_FV.npy')
    save_array(target_arr, 'bulk_event_magnitude_phase_nwf_FV_target.npy')

X = data_arr
y = target_arr
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
