import json
import os
import math
import librosa

DATASET_PATH = "./../../data/archive/Data/genres_original"
JSON_PATH = "./../../data/gtzan_mfcc_json.json"

SAMPLE_RATE = 22050
TRACK_DURATION = 29  # measured in seconds
SAMPLES_PER_TRACK = SAMPLE_RATE * TRACK_DURATION


def signal_variables(samples_per_track, num_segments, hop_length):
    """
    Perform some calculations to be used later during MFCC extraction
    :param samples_per_track: Total number of samples of the track, determined by the sampling rate and track duration
    :param num_segments: Number of segments we want to divide sample tracks into
    :param hop_length: Sliding window for FFT. Measured in # of samples
    :return: Samples per segment, and the MFCC vectors per segment
    """
    samples_per_segment = int(samples_per_track / num_segments)
    num_mfcc_vectors_per_segment = math.ceil(samples_per_segment / hop_length)

    return samples_per_segment, num_mfcc_vectors_per_segment


def save_mfcc(dataset_path, json_path, num_mfcc=13, n_fft=2048, hop_length=512, num_segments=5, subfolders=True,
              save_with_fname=False):
    """Extracts MFCCs from music dataset and saves them into a json file along witgh genre labels.
        :param subfolders: Whether to look for wav files within all subfolders of dataset_path, or just the root
        :param dataset_path (str): Path to dataset
        :param json_path (str): Path to json file used to save MFCCs
        :param num_mfcc (int): Number of coefficients to extract
        :param n_fft (int): Interval we consider to apply FFT. Measured in # of samples
        :param hop_length (int): Sliding window for FFT. Measured in # of samples
        :param: num_segments (int): Number of segments we want to divide sample tracks into
        :param save_with_fname: Save the name of the song in the 'label', if True
        :return:
        """
    if os.path.exists(json_path):
        return
    # dictionary to store mapping, labels, and MFCCs
    data = {
        "mapping": [],
        "labels": [],
        "mfcc": []
    }

    samples_per_segment, num_mfcc_vectors_per_segment = signal_variables(samples_per_track=SAMPLES_PER_TRACK,
                                                                         num_segments=num_segments,
                                                                         hop_length=hop_length)

    # loop through all genre sub-folder
    for i, (dirpath, dirnames, filenames) in enumerate(os.walk(dataset_path)):

        # ensure we're processing a genre sub-folder level
        if dirpath is not dataset_path or subfolders is False:

            # save genre label (i.e., sub-folder name) in the mapping
            semantic_label = dirpath.split('/')[-1]
            data["mapping"].append(semantic_label)
            print("\nProcessing: {}".format(semantic_label))

            # process all audio files in genre sub-dir
            for f in filenames:

                # load audio file
                file_path = os.path.join(dirpath, f)
                signal, sample_rate = librosa.load(file_path, sr=SAMPLE_RATE)

                # process all segments of audio file
                for d in range(num_segments):

                    # calculate start and finish sample for current segment
                    start = samples_per_segment * d
                    finish = start + samples_per_segment

                    # extract mfcc
                    mfcc = librosa.feature.mfcc(signal[start:finish], sample_rate, n_mfcc=num_mfcc, n_fft=n_fft,
                                                hop_length=hop_length)
                    mfcc = mfcc.T
                    # store only mfcc feature with expected number of vectors
                    if len(mfcc) == num_mfcc_vectors_per_segment:
                        data["mfcc"].append(mfcc.tolist())
                        if save_with_fname is True:
                            data["labels"].append(file_path.split('/')[-1])
                        else:
                            data["labels"].append(i - 1)
                        # print("{}, segment:{}".format(file_path, d + 1))

    # save MFCCs to json file
    with open(json_path, "w") as fp:
        json.dump(data, fp, indent=4)


if __name__ == "__main__":
    save_mfcc(DATASET_PATH, JSON_PATH, num_segments=10)
