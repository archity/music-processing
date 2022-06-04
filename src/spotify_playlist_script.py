import os

import pandas as pd
from tensorflow.keras.models import load_model

from extras.spotify_creds import client_id, client_secret
from src.utils.mfcc_extractor import save_mfcc
from src.utils.mp3_to_wav import convert_mp3_to_wav
from src.utils.neural_network_trainer import load_data, split_dataset, simple_nn_model, compile_and_train,\
    plot_results, process_predictions
from src.utils.playlist_extractor import request_playlist, download_track_preview

out_dir = "./../data/playlist-tracks/"
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

    # 3. Extract MFCCs from all the converted wav tracks
    save_mfcc(dataset_path=wav_dir, json_path=wav_dir + "../" + "playlist_songs_mfcc_json.json",
              num_segments=10, subfolders=False, save_with_fname=True)

    # 4. Load the saved MFCCs
    X, y = load_data(json_path=wav_dir + "../" + "playlist_songs_mfcc_json.json")

    # 5. Train the model on sample dataset (if not already trained)
    model_path = "./../model/saved_model_basic.h5"
    if not os.path.exists(model_path):
        X, y = load_data(json_path="./../data/gtzan_mfcc_json.json")
        train_data, validation_data = split_dataset(X=X, y=y)
        model = simple_nn_model(X)
        history = compile_and_train(model, train_data=train_data, validation_data=validation_data, name="basic",
                                    training_dump_path="./../model/")
        plot_results(history=history, name="basic", out_path="./../../img/")
    else:
        model = load_model(filepath=model_path)

    print(f"Test input shape: {X[:, :, :].shape}")

    # 6. Test the model on the converted wav songs
    predictions = model.predict(x=X[:, :, :], batch_size=32)
    genre_list, song_name_list = process_predictions(predictions, y, segments_per_track=10)
    result_dict = {"Song": song_name_list, "Genre Index": genre_list}
    df = pd.DataFrame(result_dict)
    df.to_csv('./playlist_genre_test_results.csv')
