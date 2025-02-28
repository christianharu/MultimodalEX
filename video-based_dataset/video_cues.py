from logging import config
from external.spectre.external.av_hubert.fairseq.docs import conf
from external.spectre.external.av_hubert.fairseq.fairseq.modules.quantization.pq import em
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from zmq import device
import os
import seaborn as sns
from multi_config import Config
from cv2 import mean

from math import e
from turtle import left

from numpy import empty, zeros
from requests import head
from sympy import Min
from sklearn.preprocessing import MinMaxScaler
from skimage.restoration import denoise_tv_chambolle
from scipy.optimize import curve_fit
from peakdetect import peakdetect
from turtle import right
from tqdm import tqdm


import sys
from pathlib import Path
current_dir = Path(__file__).resolve().parent
parent_dir = current_dir.parent
sys.path.append(str(parent_dir))
from hu_Utils import get_subfolders


""""
This script is used to compute the AV values per exchange for each video in the folder.
"""


def find_exchanges_av(whos_speaking):
    exchanges_points = []

    for i in range(1, len(whos_speaking)):
        if whos_speaking['speaker_status'][i] != whos_speaking['speaker_status'][i - 1]:
            exchanges_points.append(i)

    # make sure the exchanges are even
    if len(exchanges_points) % 2 != 0:
        print("The number of exchanges is not even.")
    
    num_exchanges = int(len(exchanges_points) / 2)
    print(f"Number of exchanges: {num_exchanges}")

    exchanges = pd.DataFrame(columns=['speaker', 'start', 'end', 'mean_a', 'mean_v', 'mean_derivative_a', 'mean_derivative_v']) 
    empty_rows = pd.DataFrame({col: np.nan for col in exchanges.columns}, index=range(num_exchanges))
    exchanges = pd.concat([exchanges, empty_rows], ignore_index=True)

    # add the speaker status
    counter = 0
    for i in range(len(exchanges_points)):
        # if the index is odd, add start frame
        if i % 2 == 0:
            exchanges.loc[counter, 'speaker'] = whos_speaking['speaker_status'][exchanges_points[i]]
            exchanges.loc[counter, 'start'] = exchanges_points[i] + 20

        if i % 2 == 1:
            exchanges.loc[counter, 'end'] = exchanges_points[i] + 20
            counter += 1

    for i in range(len(exchanges)):
        print(exchanges.loc[i, 'speaker'], exchanges.loc[i, 'start'], exchanges.loc[i, 'end'])

    print('\n\n')

    return exchanges



def step_function(x, a, b, c):

    heavi = np.heaviside(x - b, 0)
    result = zeros(len(x))

    # mask  the 1 values in the heaviside function
    for i in range(len(heavi)):
        if heavi[i] == 1:
            result[i] = a * heavi[i] + c
        else:
            result[i] = -2

    return result

    # return a * np.heaviside(x - b, 0) + c

def fit_step_function(data):
    clean_data = data.dropna()
    x = np.arange(len(clean_data))
    if len(x) > 0:  # Ensure there is data to fit
        popt, _ = curve_fit(step_function, x, clean_data)
        return step_function(np.arange(len(data)), *popt)
    else:
        return np.array([])  # Return an empty array if input data is empty
    

def plot_denoised(left_av, right_av, left_av_filtered, right_av_filtered):
    # plot the filtered data for the left speaker
    plt.figure(figsize=(10, 6))
    plt.plot(left_av['valence'], label='Original Data')
    plt.plot(left_av_filtered['valence'], label='Filtered Data')
    plt.xlabel('Frame')
    plt.ylabel('Left Speaker Valence')
    plt.legend()
    plt.show()

    # plot the filtered data for the right speaker
    plt.figure(figsize=(10, 6))
    plt.plot(right_av['valence'], label='Original Data')
    plt.plot(right_av_filtered['valence'], label='Filtered Data')
    plt.xlabel('Frame')
    plt.ylabel('Right Speaker Valence')
    plt.legend()
    plt.show()



def denoise_av(left_av, right_av, weight=0.3):
    # 1) use a denoise tv chambolle filter to filter the data
    left_av_filtered = left_av.copy()
    right_av_filtered = right_av.copy()
    left_av_filtered['arousal'] = denoise_tv_chambolle(left_av_filtered['arousal'], weight=weight)
    left_av_filtered['valence'] = denoise_tv_chambolle(left_av_filtered['valence'], weight=weight)
    right_av_filtered['arousal'] = denoise_tv_chambolle(right_av_filtered['arousal'], weight=weight)
    right_av_filtered['valence'] = denoise_tv_chambolle(right_av_filtered['valence'], weight=weight)


    # normalize the data for the left and right speakers in the range [-1, 1]
    scaler = MinMaxScaler(feature_range=(-1, 1))
    left_av_filtered['arousal'] = scaler.fit_transform(left_av_filtered['arousal'].values.reshape(-1, 1))
    left_av_filtered['valence'] = scaler.fit_transform(left_av_filtered['valence'].values.reshape(-1, 1))
    right_av_filtered['arousal'] = scaler.fit_transform(right_av_filtered['arousal'].values.reshape(-1, 1))
    right_av_filtered['valence'] = scaler.fit_transform(right_av_filtered['valence'].values.reshape(-1, 1))


    # 1.1) plot the denoised data and the original data for the left and right speakers
    # plot_denoised(left_av, right_av, left_av_filtered, right_av_filtered)

    return left_av_filtered, right_av_filtered


def plot_peaks(left_val_peaks_data, left_arousal_peaks_data, right_val_peaks_data, right_arousal_peaks_data):

    # 

    # plot the peaks for the left speaker valence
    plt.figure(figsize=(10, 6))
    for i in range(len(left_val_peaks_data)):
        plt.plot(left_val_peaks_data[i])
    plt.xlabel('Frame')
    plt.ylabel('Left Speaker Valence Peaks')
    plt.show()

    # plot the peaks for the left speaker arousal
    plt.figure(figsize=(10, 6))
    for i in range(len(left_arousal_peaks_data)):
        plt.plot(left_arousal_peaks_data[i])
    plt.xlabel('Frame')
    plt.ylabel('Left Speaker Arousal Peaks')
    plt.show()

    # plot the peaks for the right speaker valence
    plt.figure(figsize=(10, 6))
    for i in range(len(right_val_peaks_data)):
        plt.plot(right_val_peaks_data[i])
    plt.xlabel('Frame')
    plt.ylabel('Right Speaker Valence Peaks')
    plt.show()

    # plot the peaks for the right speaker arousal
    plt.figure(figsize=(10, 6))
    for i in range(len(right_arousal_peaks_data)):
        plt.plot(right_arousal_peaks_data[i])
    plt.xlabel('Frame')
    plt.ylabel('Right Speaker Arousal Peaks')
    plt.show()


def find_peaks(data):
    max_peaks, min_peaks = peakdetect(data, lookahead=20)
    max_peaks = np.array(max_peaks)
    min_peaks = np.array(min_peaks)
    return max_peaks, min_peaks


def get_peak_values(left_av, right_av, exchanges, window=10):
    left_val_max_peaks, left_val_min_peaks = find_peaks(left_av['valence'])
    right_val_max_peaks, right_val_min_peaks = find_peaks(right_av['valence'])

    # find the peaks for the left and right speakers arousal
    left_arousal_max_peaks, left_arousal_min_peaks = find_peaks(left_av['arousal'])
    right_arousal_max_peaks, right_arousal_min_peaks = find_peaks(right_av['arousal'])

    # remove the peaks that are not in the exchanges
    right_peaks_valance = np.empty((0, right_val_max_peaks.shape[1]), dtype=right_val_max_peaks.dtype)
    right_peaks_arousal = np.empty((0, right_val_max_peaks.shape[1]), dtype=right_val_max_peaks.dtype)
    left_peaks_valance = np.empty((0, right_val_max_peaks.shape[1]), dtype=right_val_max_peaks.dtype)
    left_peaks_arousal = np.empty((0, right_val_max_peaks.shape[1]), dtype=right_val_max_peaks.dtype)

    for i in range(len(exchanges)):
        start = exchanges.loc[i, 'start']
        end = exchanges.loc[i, 'end']
        speaker = exchanges.loc[i, 'speaker']

        if speaker == 'left':
            filtered_elements = left_val_max_peaks[(left_val_max_peaks[:, 0] > start) & (left_val_max_peaks[:, 0] < end)][:, 0]
            filtered_elements = filtered_elements.reshape(-1, 1).flatten().astype(int)
            filtered_rows = left_val_max_peaks[np.isin(left_val_max_peaks[:, 0], filtered_elements)]
            left_peaks_valance = np.append(left_peaks_valance, filtered_rows, axis=0)

            filtered_elements = left_val_min_peaks[(left_val_min_peaks[:, 0] > start) & (left_val_min_peaks[:, 0] < end)][:, 0]
            if len(filtered_elements) > 1:
                filtered_elements = filtered_elements[:-1]


            filtered_elements = filtered_elements.reshape(-1, 1).flatten().astype(int)
            filtered_rows = left_val_min_peaks[np.isin(left_val_min_peaks[:, 0], filtered_elements)]
            left_peaks_valance = np.append(left_peaks_valance, filtered_rows, axis=0)

            filtered_elements = left_arousal_max_peaks[(left_arousal_max_peaks[:, 0] > start) & (left_arousal_max_peaks[:, 0] < end)][:, 0]
            filtered_elements = filtered_elements.reshape(-1, 1).flatten().astype(int)
            filtered_rows = left_arousal_max_peaks[np.isin(left_arousal_max_peaks[:, 0], filtered_elements)]
            left_peaks_arousal = np.append(left_peaks_arousal, filtered_rows, axis=0)

            filtered_elements = left_arousal_min_peaks[(left_arousal_min_peaks[:, 0] > start) & (left_arousal_min_peaks[:, 0] < end)][:, 0]
            if len(filtered_elements) > 1:
                filtered_elements = filtered_elements[:-1]


            filtered_elements = filtered_elements.reshape(-1, 1).flatten().astype(int)
            filtered_rows = left_arousal_min_peaks[np.isin(left_arousal_min_peaks[:, 0], filtered_elements)]
            left_peaks_arousal = np.append(left_peaks_arousal, filtered_rows, axis=0)

        elif speaker == 'right':
            filtered_elements = right_val_max_peaks[(right_val_max_peaks[:, 0] > start) & (right_val_max_peaks[:, 0] < end)][:, 0]
            filtered_elements = filtered_elements.reshape(-1, 1).flatten().astype(int)
            filtered_rows = right_val_max_peaks[np.isin(right_val_max_peaks[:, 0], filtered_elements)]
            right_peaks_valance = np.append(right_peaks_valance, filtered_rows, axis=0)

            filtered_elements = right_val_min_peaks[(right_val_min_peaks[:, 0] > start) & (right_val_min_peaks[:, 0] < end)][:, 0]
            if len(filtered_elements) > 1:
                filtered_elements = filtered_elements[:-1]

            filtered_elements = filtered_elements.reshape(-1, 1).flatten().astype(int)
            filtered_rows = right_val_min_peaks[np.isin(right_val_min_peaks[:, 0], filtered_elements)]
            right_peaks_valance = np.append(right_peaks_valance, filtered_rows, axis=0)

            filtered_elements = right_arousal_max_peaks[(right_arousal_max_peaks[:, 0] > start) & (right_arousal_max_peaks[:, 0] < end)][:, 0]
            filtered_elements = filtered_elements.reshape(-1, 1).flatten().astype(int)
            filtered_rows = right_arousal_max_peaks[np.isin(right_arousal_max_peaks[:, 0], filtered_elements)]
            right_peaks_arousal = np.append(right_peaks_arousal, filtered_rows, axis=0)

            filtered_elements = right_arousal_min_peaks[(right_arousal_min_peaks[:, 0] > start) & (right_arousal_min_peaks[:, 0] < end)][:, 0]
            if len(filtered_elements) > 1:
                filtered_elements = filtered_elements[:-1]
                
            filtered_elements = filtered_elements.reshape(-1, 1).flatten().astype(int)
            filtered_rows = right_arousal_min_peaks[np.isin(right_arousal_min_peaks[:, 0], filtered_elements)]
            right_peaks_arousal = np.append(right_peaks_arousal, filtered_rows, axis=0)


    left_peaks_valance = np.vstack(left_peaks_valance)
    left_peaks_arousal = np.vstack(left_peaks_arousal)
    right_peaks_valance = np.vstack(right_peaks_valance)
    right_peaks_arousal = np.vstack(right_peaks_arousal)

    left_val_peaks_data = []
    left_arousal_peaks_data = []
    right_val_peaks_data = []
    right_arousal_peaks_data = []
    
    # left speaker valence peaks
    for i in range(len(left_peaks_valance)):
        left_val_peaks_data.append(left_av['valence'][int(left_peaks_valance[i][0]) - window:int(left_peaks_valance[i][0]) + window])

    # left speaker arousal peaks
    for i in range(len(left_peaks_arousal)):
        left_arousal_peaks_data.append(left_av['arousal'][int(left_peaks_arousal[i][0]) - window:int(left_peaks_arousal[i][0]) + window])
        

    # right speaker valence peaks
    for i in range(len(right_peaks_valance)):
        right_val_peaks_data.append(right_av['valence'][int(right_peaks_valance[i][0]) - window:int(right_peaks_valance[i][0]) + window])

    # right speaker arousal peaks
    for i in range(len(right_peaks_arousal)):
        right_arousal_peaks_data.append(right_av['arousal'][int(right_peaks_arousal[i][0]) - window:int(right_peaks_arousal[i][0]) + window])

    
    # plot_peaks(left_val_peaks_data, left_arousal_peaks_data, right_val_peaks_data, right_arousal_peaks_data)

    return left_val_peaks_data, left_arousal_peaks_data, right_val_peaks_data, right_arousal_peaks_data



def plot_fitted_peaks(left_val_peaks_fitted, left_aroural_peaks_fitted, right_val_peaks_fitted, right_arousal_peaks_fitted, exchanges=None):
    
        # plot the fitted peaks for the left speaker valence
        plt.figure(figsize=(10, 6))
        for i in range(len(left_val_peaks_fitted)):
            index, data = zip(*left_val_peaks_fitted[i][1])
            plt.plot(index, data)


        # plot the exchanges
        for i in range(len(exchanges)):
            plt.axvline(x=exchanges.loc[i, 'start'], color='r')
            plt.axvline(x=exchanges.loc[i, 'end'], color
            ='r')


        plt.xlabel('Frame')
        plt.ylabel('Left Speaker Valence')
        plt.legend()
        plt.show()
    
        # plot the fitted peaks for the left speaker arousal
        plt.figure(figsize=(10, 6))
        for i in range(len(left_aroural_peaks_fitted)):
            index, data = zip(*left_aroural_peaks_fitted[i][1])
            plt.plot(index, data)
        plt.xlabel('Frame')
        plt.ylabel('Left Speaker Arousal Peaks')
        plt.show()
    
        # plot the fitted peaks for the right speaker valence
        plt.figure(figsize=(10, 6))
        for i in range(len(right_val_peaks_fitted)):
            index, data = zip(*right_val_peaks_fitted[i][1])
            plt.plot(index, data)
        plt.xlabel('Frame')
        plt.ylabel('Right Speaker Valence Peaks')
        plt.show()
    
        # plot the fitted peaks for the right speaker arousal
        plt.figure(figsize=(10, 6))
        for i in range(len(right_arousal_peaks_fitted)):
            index, data = zip(*right_arousal_peaks_fitted[i][1])
            plt.plot(index, data)
        plt.xlabel('Frame')
        plt.ylabel('Right Speaker Arousal Peaks')
        plt.show()


def return_step_function(left_val_peaks_data, left_arousal_peaks_data, right_val_peaks_data, right_arousal_peaks_data, exchanges):

    left_val_peaks_fitted = []
    left_aroural_peaks_fitted = []
    right_val_peaks_fitted = []
    right_arousal_peaks_fitted = []


    # left speaker valence fitted peaks
    for index, data in enumerate(left_val_peaks_data):
        fitted_data = fit_step_function(data)
        fitted_data_with_indices = list(zip(data.index, fitted_data))
        left_val_peaks_fitted.append((index, fitted_data_with_indices))

    # left speaker arousal fitted peaks
    for index, data in enumerate(left_arousal_peaks_data):
        fitted_data = fit_step_function(data)
        fitted_data_with_indices = list(zip(data.index, fitted_data))
        left_aroural_peaks_fitted.append((index, fitted_data_with_indices))

    # right speaker valence fitted peaks
    for index, data in enumerate(right_val_peaks_data):
        fitted_data = fit_step_function(data)
        fitted_data_with_indices = list(zip(data.index, fitted_data))
        right_val_peaks_fitted.append((index, fitted_data_with_indices))

    # right speaker arousal fitted peaks
    for index, data in enumerate(right_arousal_peaks_data):
        fitted_data = fit_step_function(data)
        fitted_data_with_indices = list(zip(data.index, fitted_data))
        right_arousal_peaks_fitted.append((index, fitted_data_with_indices))


    # 3.3) replace the -2 values with None in the fitted data
    for i in range(len(left_val_peaks_fitted)):
        left_val_peaks_fitted[i] = (left_val_peaks_fitted[i][0], [(x, y) if y != -2 else (x, None) for x, y in left_val_peaks_fitted[i][1]])

    for i in range(len(left_aroural_peaks_fitted)):
        left_aroural_peaks_fitted[i] = (left_aroural_peaks_fitted[i][0], [(x, y) if y != -2 else (x, None) for x, y in left_aroural_peaks_fitted[i][1]])

    for i in range(len(right_val_peaks_fitted)):
        right_val_peaks_fitted[i] = (right_val_peaks_fitted[i][0], [(x, y) if y != -2 else (x, None) for x, y in right_val_peaks_fitted[i][1]])

    for i in range(len(right_arousal_peaks_fitted)):
        right_arousal_peaks_fitted[i] = (right_arousal_peaks_fitted[i][0], [(x, y) if y != -2 else (x, None) for x, y in right_arousal_peaks_fitted[i][1]])



    # plot_fitted_peaks(left_val_peaks_fitted, left_aroural_peaks_fitted, right_val_peaks_fitted, right_arousal_peaks_fitted, exchanges)

    return left_val_peaks_fitted, left_aroural_peaks_fitted, right_val_peaks_fitted, right_arousal_peaks_fitted



def compute_mean_av(left_val_peaks_fitted, left_aroural_peaks_fitted, right_val_peaks_fitted, right_arousal_peaks_fitted, exchanges):

    for i in range(len(exchanges)):
        start = exchanges.loc[i, 'start']
        end = exchanges.loc[i, 'end']
        speaker = exchanges.loc[i, 'speaker']
        mean_left_v, mean_left_a, mean_right_v, mean_right_a = 0, 0, 0, 0

        if speaker == 'left':
            for j in range(len(left_val_peaks_fitted)):
                if left_val_peaks_fitted[j][1]:  # Check if there is data to unpack
                    index, data = zip(*left_val_peaks_fitted[j][1])
                    for k in range(len(data)):
                        if index[k] >= start and index[k] <= end:
                            mean_left_v += 0 if data[k] is None else data[k]
                else:
                    continue  # Skip this iteration if there's nothing to unpack
            mean_left_v /= max(1, end - start)  # Avoid division by zero
            exchanges.loc[i, 'mean_v'] = mean_left_v

            for j in range(len(left_aroural_peaks_fitted)):
                if left_aroural_peaks_fitted[j][1]:  # Check if there is data to unpack
                    index, data = zip(*left_aroural_peaks_fitted[j][1])
                    for k in range(len(data)):
                        if index[k] >= start and index[k] <= end:
                            mean_left_a += 0 if data[k] is None else data[k]
                else:
                    continue  # Skip this iteration if there's nothing to unpack
            mean_left_a /= max(1, end - start)  # Avoid division by zero
            exchanges.loc[i, 'mean_a'] = mean_left_a

        elif speaker == 'right':
            for j in range(len(right_val_peaks_fitted)):
                if right_val_peaks_fitted[j][1]:
                    index, data = zip(*right_val_peaks_fitted[j][1])
                    for k in range(len(data)):
                        if index[k] >= start and index[k] <= end:
                            mean_right_v += 0 if data[k] is None else data[k]
                else:
                    continue
            mean_right_v /= max(1, end - start)
            exchanges.loc[i, 'mean_v'] = mean_right_v


            for j in range(len(right_arousal_peaks_fitted)):
                if right_arousal_peaks_fitted[j][1]:  # Check if there is data to unpack
                    index, data = zip(*right_arousal_peaks_fitted[j][1])
                    for k in range(len(data)):
                        if index[k] >= start and index[k] <= end:
                            mean_right_a += 0 if data[k] is None else data[k]
                else:
                    continue  # Skip this iteration if there's nothing to unpack
            mean_right_a /= max(1, end - start)  # Avoid division by zero
            exchanges.loc[i, 'mean_a'] = mean_right_a


    # change the dataframe to a new format with five columns
    exchanges_new = pd.DataFrame(columns=['exchange', 'arousal_right', 'valence_right', 'arousal_left', 'valence_left'])

    num_exchanges = int(len(exchanges) / 2)

    empty_rows = pd.DataFrame({col: np.nan for col in exchanges_new.columns}, index=range(num_exchanges))
    exchanges_new = pd.concat([exchanges_new, empty_rows], ignore_index=True)

    exchange_counter = 0
    for i in range(len(exchanges)):
        exchanges_new.loc[exchange_counter, 'exchange'] = exchange_counter

        if exchanges.loc[i, 'speaker'] == 'right':
            exchanges_new.loc[exchange_counter, 'arousal_right'] = exchanges.loc[i, 'mean_a']
            exchanges_new.loc[exchange_counter, 'valence_right'] = exchanges.loc[i, 'mean_v']

        elif exchanges.loc[i, 'speaker'] == 'left':
            exchanges_new.loc[exchange_counter, 'arousal_left'] = exchanges.loc[i, 'mean_a']
            exchanges_new.loc[exchange_counter, 'valence_left'] = exchanges.loc[i, 'mean_v']
            exchange_counter += 1

    return exchanges_new



#### export the exchanges for each video in the folder ####
def export_exchanges_av(folder, out, name_base):
    base_path = folder
    out_path = os.path.join(base_path, out)

    # create the output folder
    if not os.path.exists(out_path):
        os.makedirs(out_path)

    # load speaker status files for each video
    # get a list of all the files in the folder


    folder_status = os.path.join(base_path, 'speaker_status')
    status_files = [name for name in os.listdir(folder_status) if os.path.isfile(os.path.join(folder_status, name))]


    # get a list of files in the av_per_frame folder
    folder_av = os.path.join(base_path, 'av_per_frame')
    av_files = [name for name in os.listdir(folder_av) if os.path.isfile(os.path.join(folder_av, name))]


    # # keep only the status files that caontain the name_base
    # status_files = [folder for folder in status_files if name_base in folder]
    # num_files = len(status_files)


    # load av data for each video
    for current_file in tqdm(status_files):
        # remove 'speaker_status' from the name
        current_file = current_file.replace('_speaker_status.csv', '')
        print(f"Processing {current_file}...")


        # load the av data per exchange
        left_av = pd.read_csv(os.path.join(folder_av,f"{current_file}_left_av_av_per_frame.csv"))
        right_av = pd.read_csv(os.path.join(folder_av,f"{current_file}_right_av_av_per_frame.csv"))
        whos_speaking = pd.read_csv(os.path.join(folder_status, f"{current_file}_speaker_status.csv"))

        # print(f"Left AV shape: {left_av.shape}")
        # print(f"Right AV shape: {right_av.shape}")

        # find the exchanges
        exchanges = find_exchanges_av(whos_speaking)

        if len(exchanges) == 0:
            # save the file name to a text file
            with open(os.path.join(out_path, "files_with_no_exchanges.txt"), "a") as f:
                f.write(f"{current_file}\n")
            continue

        # denoise the av data
        left_av, right_av = denoise_av(left_av, right_av, 0.3)

        # get the peak values
        left_val_peaks_data, left_arousal_peaks_data, right_val_peaks_data, right_arousal_peaks_data = get_peak_values(left_av, right_av, exchanges, window = 15)

        if len(left_val_peaks_data) == 0 or len(left_arousal_peaks_data) == 0 or len(right_val_peaks_data) == 0 or len(right_arousal_peaks_data) == 0:
            # save the file name to a text file
            with open(os.path.join(out_path, "files_with_no_peaks.txt"), "a") as f:
                f.write(f"{current_file}\n")
            continue

        # fit the step function
        left_val_peaks_fitted, left_aroural_peaks_fitted, right_val_peaks_fitted, right_arousal_peaks_fitted = return_step_function(left_val_peaks_data, left_arousal_peaks_data, right_val_peaks_data, right_arousal_peaks_data, exchanges)


        # get mean values for each exchange
        av_exchanges = compute_mean_av(left_val_peaks_fitted, left_aroural_peaks_fitted, right_val_peaks_fitted, right_arousal_peaks_fitted, exchanges)
        print(av_exchanges.head())

        # export the exchanges to a csv file
        av_exchanges.to_csv(os.path.join(out_path, f"{current_file}_av_exchanges.csv"), index=False)





if __name__ == '__main__':
    config = Config()

    config.add_argument(
        "--output",
        type=str,
        default="av_mimicry",
        help="Path to the output folder",
    )

    config.add_argument(
        "--name_base",
        type=str,
        default="subject",
        help="Base name of the video cues",
    )

    args = config.args

    folder = args.folder
    output = args.output
    name_base = args.name_base

    # get current working directory
    working_dir = os.getcwd()
    folder = os.path.join(working_dir, 'Groups' ,'Dataset')
    name_base = 'lottery'
    output =  'av_per_exchange_15'

    # compute the AV values for each video in the folder
    export_exchanges_av(folder, output, name_base)
