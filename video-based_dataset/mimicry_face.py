import pandas as pd
import numpy as np
import os
import argparse
from tqdm import tqdm

import sys
from pathlib import Path
current_dir = Path(__file__).resolve().parent
parent_dir = current_dir.parent
sys.path.append(str(parent_dir))


""""
This script is used to compute the mimicry values per exchange for each video in the folder.
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


def find_mimicry(face_speaker, face_listener, pose_speaker, pose_listener, exchanges):  
    # cross-correlation between the facial expressions of the speaker and the listener
    face_speaker_derivative = np.diff(face_speaker, axis=0)
    face_listener_derivative = np.diff(face_listener, axis=0)

    # compute the derivatives of the pose
    pose_speaker_derivative = np.diff(pose_speaker[:, :, 1:2], axis=0)
    pose_listener_derivative = np.diff(pose_listener[:, :, 1:2], axis=0)


    lag_face = []
    lag_pose = []

    window_face = 80
    window_pose = 80

    num_exchanges = len(exchanges)
    similarity_face = [[] for _ in range(num_exchanges)]
    similarity_pose = [[] for _ in range(num_exchanges)]
    cross_correlation_face = [[] for _ in range(num_exchanges)]
    cross_correlation_pose = [[] for _ in range(num_exchanges)]

    for i in range(0, num_exchanges):
        # range for the speaker
        speaker_start = exchanges.loc[i, 'start']
        speaker_end = exchanges.loc[i, 'end']

        # range for the listener
        listener_start = exchanges.loc[i, 'start']
        listener_end = exchanges.loc[i, 'end']

        # get the cosine similarity between the derivatives of the pose and facial expressions
        speaker_face_derivative = face_speaker_derivative[speaker_start:speaker_end]
        listener_face_derivative = face_listener_derivative[listener_start:listener_end]
        speaker_pose_derivative = pose_speaker_derivative[speaker_start:speaker_end]
        listener_pose_derivative = pose_listener_derivative[listener_start:listener_end]


        # get the similaraty at frame interval
        for j in range(0, len(speaker_face_derivative), window_face):
            speaker_face_derivative_interval = speaker_face_derivative[j:j+window_face]
            listener_face_derivative_interval = listener_face_derivative[j:j+window_face]

        for j in range(0, len(speaker_pose_derivative), window_pose):
            speaker_pose_derivative_interval = speaker_pose_derivative[j:j+window_pose]
            listener_pose_derivative_interval = listener_pose_derivative[j:j+window_pose]

            # flatten the arrays
            speaker_face_derivative_interval = speaker_face_derivative_interval.flatten()
            listener_face_derivative_interval = listener_face_derivative_interval.flatten()
            speaker_pose_derivative_interval = speaker_pose_derivative_interval.flatten()
            listener_pose_derivative_interval = listener_pose_derivative_interval.flatten()

            # normalize the arrays
            speaker_face_derivative_interval = speaker_face_derivative_interval / np.linalg.norm(speaker_face_derivative_interval)
            listener_face_derivative_interval = listener_face_derivative_interval / np.linalg.norm(listener_face_derivative_interval)
            speaker_pose_derivative_interval = speaker_pose_derivative_interval / np.linalg.norm(speaker_pose_derivative_interval)
            listener_pose_derivative_interval = listener_pose_derivative_interval / np.linalg.norm(listener_pose_derivative_interval)
            

            # find the cross-correlation between the facial expressions and pose
            correlation_face = np.correlate(speaker_face_derivative_interval, listener_face_derivative_interval, mode='full')
            norm_face_a = np.linalg.norm(speaker_face_derivative_interval)
            norm_face_v = np.linalg.norm(listener_face_derivative_interval)
            normalized_face = correlation_face / (norm_face_a * norm_face_v)

            correlation_pose = np.correlate(speaker_pose_derivative_interval, listener_pose_derivative_interval, mode='full')
            norm_pose_a = np.linalg.norm(speaker_pose_derivative_interval)
            norm_pose_v = np.linalg.norm(listener_pose_derivative_interval)
            normalized_pose = correlation_pose / (norm_pose_a * norm_pose_v)

            # find the lag
            lag_face.append(np.argmax(normalized_face) - len(speaker_face_derivative_interval) + 1)
            lag_pose.append(np.argmax(normalized_pose) - len(speaker_pose_derivative_interval) + 1)


            cross_correlation_face[i].append(np.max(correlation_face))
            cross_correlation_pose[i].append(np.max(correlation_pose))


            # shift signals using the lag and find the cosine similarity
            speaker_face_derivative_interval = np.roll(speaker_face_derivative_interval, lag_face[-1])

            listener_face_derivative_interval = listener_face_derivative_interval
            speaker_pose_derivative_interval = np.roll(speaker_pose_derivative_interval, lag_face[-1])

            listener_pose_derivative_interval = np.roll(listener_pose_derivative_interval, lag_pose[-1])
            speaker_pose_derivative_interval = np.roll(speaker_pose_derivative_interval, lag_pose[-1])


            # cosine similarity between the facial and pose derivatives
            # only append if i is not an odd number
            similarity_face[i].append(np.dot(speaker_face_derivative_interval.T, listener_face_derivative_interval) / (np.linalg.norm(speaker_face_derivative_interval) * np.linalg.norm(listener_face_derivative_interval)))

            similarity_pose[i].append(np.dot(speaker_pose_derivative_interval.T, listener_pose_derivative_interval) / (np.linalg.norm(speaker_pose_derivative_interval) * np.linalg.norm(listener_pose_derivative_interval)))


    return similarity_face, similarity_pose



def _load_status_files(base_path, name_base=""):
    folder_status = os.path.join(base_path, 'speaker_status')
    status_files = [name for name in os.listdir(folder_status) if os.path.isfile(os.path.join(folder_status, name))]
    if name_base:
        status_files = [name for name in status_files if name_base in name]
    return sorted(status_files)


def compute_mimicry_scores_for_file(base_path, current_file):
    folder_status = os.path.join(base_path, 'speaker_status')
    folder_av = os.path.join(base_path, 'av_frames')

    face_speaker = np.load(os.path.join(folder_av, f"{current_file}_right_av_face_data.npy"))
    pose_speaker = np.load(os.path.join(folder_av, f"{current_file}_right_av_pose_data.npy"))
    face_listener = np.load(os.path.join(folder_av, f"{current_file}_left_av_face_data.npy"))
    pose_listener = np.load(os.path.join(folder_av, f"{current_file}_left_av_pose_data.npy"))
    whos_speaking = pd.read_csv(os.path.join(folder_status, f"{current_file}_speaker_status.csv"))

    # Keep only the first detected face/body track.
    face_speaker = face_speaker[:, 0:1, :]
    face_listener = face_listener[:, 0:1, :]
    pose_speaker = pose_speaker[:, 0:1, :]
    pose_listener = pose_listener[:, 0:1, :]

    exchanges = find_exchanges_av(whos_speaking)
    exchanges = exchanges[exchanges['speaker'] == 'right'].reset_index(drop=True)

    similarity_face, similarity_pose = find_mimicry(
        face_speaker,
        face_listener,
        pose_speaker,
        pose_listener,
        exchanges,
    )

    mean_mimicry_face = np.nan_to_num([np.mean(similarity_face[i]) for i in range(len(exchanges))])
    mean_mimicry_pose = np.nan_to_num([np.mean(similarity_pose[i]) for i in range(len(exchanges))])
    exchange_id = np.arange(0, len(exchanges))

    return pd.DataFrame(
        {
            'exchange': exchange_id,
            'mean_mimicry_face': mean_mimicry_face,
            'mean_mimicry_pose': mean_mimicry_pose,
        }
    )


def threshold_mimicry_scores(scores_df, threshold_face):
    return pd.DataFrame(
        {
            'exchange': scores_df['exchange'].astype(int),
            'mimicry_face': np.where(scores_df['mean_mimicry_face'] > threshold_face, 1, 0),
            'mimicry_pose': np.where(scores_df['mean_mimicry_pose'] > threshold_face, 1, 0),
        }
    )


def export_mimicry_face(folder, out, name_base, threshold_face):
    base_path = folder
    out_path = os.path.join(base_path, out)
    os.makedirs(out_path, exist_ok=True)

    for status_file in tqdm(_load_status_files(base_path, name_base)):
        current_file = status_file.replace('_speaker_status.csv', '')
        print(f"Processing {current_file}...")
        scores_df = compute_mimicry_scores_for_file(base_path, current_file)
        mimicry = threshold_mimicry_scores(scores_df, threshold_face)
        mimicry.to_csv(os.path.join(out_path, f"{current_file}_mimicry.csv"), index=False)





if __name__ == '__main__':
    # Create an argument parser
    parser = argparse.ArgumentParser(description="Process mimicry face arguments.")

    # Add arguments
    parser.add_argument(
        "--output",
        type=str,
        default="av_mimicry",
        help="Path to the output folder",
    )

    parser.add_argument(
        "--name_base",
        type=str,
        default="sub_13_conv_3",
        help="Base name of the video cues",
    )

    parser.add_argument(
        "--threshold_face",
        type=float,
        default=0.02,  # Corrected to a float value
        help="Threshold for the cross-correlation for facial expressions",
    )

    # Parse arguments
    args = parser.parse_args()

    # Extract arguments
    output = args.output
    name_base = args.name_base
    threshold_face = args.threshold_face

    working_dir = os.getcwd()
    folder = os.path.join(working_dir, 'video-based_dataset')

    # Compute the mimicry values per exchange
    export_mimicry_face(folder, output, name_base, threshold_face)
