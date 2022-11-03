# Music Processing

<p align="center">
<img width="350" src="./img/plot_wave_white_bg.jpg">
</p>

This repository contains work related to audio processing and music genre classifier. The documentation for each of the following sections can be found within the corresponding notebooks.

- [X] 1. Input Data Visualization
- [X] 2. MFCC Extraction
- [X] 3. Genre Classifier using GTZAN dataset
- [X] 4. Extracting & testing (predicting genre) of songs from Spotify

<br>

---

<br>

## 0. Installation

- From the root of this repository, install the top-level package `src`:

    ```commandline
    pip install -e .
    ```
- Install all the libraries in the `requirements.txt` file: 

    ```commandline
    pip install -r requirements.txt
    ```

<br>

## 1. Input Data Visualization

[Notebook](./notebooks/1-basic-input-data-visualization.ipynb)

Some basic analysis of an audio file like waveform plotting, spectrum display were performed in order to understand about audio data type

<p align="left">
<img width="250" src="./img/plot_log-spectrogram_white_bg.jpg">
</p>

* Waveform plotting
* Power Spectral Density (PSD) plot
* Spectrogram
* Mel Spectrogram
* MFCC

<br>

## 2. MFCC Extraction

[Notebook](./notebooks/2-mfcc-extractor.ipynb)

<p align="left">
<img width="500" src="./img/wave_to_mfcc.jpg">
</p>

Utility to read all the .wav files stored in separate folders according to their genre, extract MFCC, and store these values in json format.

<br>

## 3. Genre Classifier using GTZAN Dataset

[Notebook](./notebooks/3-gtzan-neural-network.ipynb)

<p align="left">
<img width="250" src="./img/train_test_plot_LSTM_white_bg.jpg">
</p>

* Neural Network
* Improved Neural Network
* Convolutional Neural Network (CNN)
* RNN - LSTM

Following 10 genres were used for training:
```py
0: "blues",
1: "classical",
2: "country",
3: "disco",
4: "hiphop",
5: "jazz",
6: "metal",
7: "pop",
8: "reggae",
9: "rock"
```

<br>

## 4. Detecting Genre of Songs from a Spotify Playlist
 
[Script](./src/spotify_playlist_script.py)

A script based pipeline script broadly involving the following steps:

1. Download a ~30sec sample of all songs from a public Spotify playlist
2. Convert the songs from `.mp3` to `.wav`
3. Extract MFCCs from the playlist's tracks
4. Extract MFCCs from the GTZAN dataset's tracks
5. Train a Neural Network based model on GTZAN dataset
6. Test and obtain results of the model on songs from Spotify playlist

Genre of a song is quite subjective, and a song can be composed of multiple genres. Instead of classifying a song into a single genre out of the 10 trained genres, the network outputs all the possible predictions of genres for each song. A song is broken into a certain number of segments (10 as chosen) and we get prediction of genre from each of the 10 segments. We simply extract the unique genres from the 10 (which usually turn out to be $\le$ 5).

Results of an LSTM based trained model on some of songs from the playlist [20th Century](https://open.spotify.com/playlist/7eYZpOTqL0Y3kwEsxNr0PI?si=457d7889fa3a4343):

| Song Name                                                                     | Predicted Genre        |
|-------------------------------------------------------------------------------|------------------------|
| [Black or White](https://en.wikipedia.org/wiki/Black_or_White)                | 3, **'4'**, **'7'**    |
| [Danger Zone](https://en.wikipedia.org/wiki/Danger_Zone_(Kenny_Loggins_song)) | 3, 4, 7, **'9'**       |
| [I Ran (So Far Away)](https://en.wikipedia.org/wiki/I_Ran_(So_Far_Away))      | 3, 4, **'9'**          |
| [Fast Car](https://en.wikipedia.org/wiki/Fast_Car)                            | 2, 4, **'7'**, 8       |
| [Take My Breadth Away](https://en.wikipedia.org/wiki/Take_My_Breath_Away)     | 2, **'3'**, **'9'**    |
| [99 Luftballons](https://en.wikipedia.org/wiki/99_Luftballons)                | 2, **'3'**, 6, **'9'** |

Indices in quotes and **bold** indicate the genre also reported by the respective song's Wikipedia page (including stylistic origins)
