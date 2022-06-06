import os

import pandas as pd
from tensorflow.keras.models import load_model

from extras.spotify_creds import client_id, client_secret
from src.utils.mfcc_extractor import save_mfcc
from src.utils.mp3_to_wav import convert_mp3_to_wav
from src.utils.neural_network_trainer import load_data, split_dataset, simple_nn_model, compile_and_train, \
    plot_results, process_predictions, lstm_model
from src.utils.playlist_extractor import request_playlist, download_track_preview

out_dir = "./../data/playlist-tracks/"
gtzan_json = "./../data/gtzan_mfcc_json.json"
path_to_ffmpeg_exe = "./../extras/ffmpeg-2022-05-26-git-0dcbe1c1aa-full_build/bin/ffmpeg.exe"
playlist_id = "spotify:playlist:7eYZpOTqL0Y3kwEsxNr0PI"

if __name__ == "__main__":

    # 1. Download playlist tracks
    sp, playlist = request_playlist(client_id=client_id, client_secret=client_secret, playlist_id=playlist_id)
    results = sp.playlist(playlist['id'], fields="tracks,next")
    tracks = results['tracks']
    mp3_dir = download_track_preview(tracks=tracks, playlist_name=playlist['name'], out_path=out_dir)

    # 2. Convert downloaded tracks from mp3 to wav
    wav_dir = convert_mp3_to_wav(input_dir=mp3_dir, ffmpeg_path=path_to_ffmpeg_exe)

    # 3. Extract MFCCs from all the converted wav tracks (test set)
    save_mfcc(dataset_path=wav_dir, json_path=wav_dir + "../" + "playlist_songs_mfcc_json.json",
              num_segments=10, subfolders=False, save_with_fname=True)

    # 4. Extract MFCCs from the GTZAN dataset (train set)
    save_mfcc(dataset_path="./../data/archive/Data/genres_original/", json_path=gtzan_json,
              num_segments=10, subfolders=True)

    # 5. Train the model on sample dataset (if not already trained)
    model_dict = {"basic": simple_nn_model, "lstm": lstm_model}
    model_type = "lstm"
    model_path = f"./../model/saved_model_{model_type}.h5"
    if not os.path.exists(model_path):
        X, y = load_data(json_path=gtzan_json)
        train_data, validation_data = split_dataset(X=X, y=y)
        model = model_dict[f"{model_type}"](X)
        history = compile_and_train(model, train_data=train_data, validation_data=validation_data, name=f"{model_type}",
                                    training_dump_path="./../model/")
        plot_results(history=history, name=f"{model_type}", out_path="./../../img/")
    else:
        model = load_model(filepath=model_path)

    # 6. Load the saved MFCCs of the playlist's wav tracks
    X, y = load_data(json_path=wav_dir + "../" + "playlist_songs_mfcc_json.json")

    # 7. Test the model on the converted wav songs
    predictions = model.predict(x=X[:, :, :], batch_size=32)
    genre_list, song_name_list = process_predictions(predictions, y, segments_per_track=10)
    result_dict = {"Song": song_name_list, "Genre Index": genre_list}
    df = pd.DataFrame(result_dict)
    df.to_csv(wav_dir + "../" + f"playlist_genre_test_results_{model_type}.csv")
