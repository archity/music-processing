# Music Processing

<p align="center">
<img width="350" src="./img/plot_wave_white_bg.jpg">
</p>

This repository contains work related to audio processing and music genre classifier. The documentation for each of the following sections can be found within the corresponding notebooks.

- [X] 1. Input Data Visualization
- [X] 2. MFCC Extraction
- [X] 3. Genre Classifier using GTZAN dataset
- [ ] 4. Extracting & testing songs from Spotify

---

## 1. Input Data Visualization

[Notebook](./1-basic-input-data-visualization.ipynb)

Some basic analysis of an audio file like waveform plotting, spectrum display were performed in order to understand about audio data type

<p align="left">
<img width="250" src="./img/plot_log-spectrogram_white_bg.jpg">
</p>

* Waveform plotting
* Power Spectral Density (PSD) plot
* Spectrogram
* Mel Spectrogram
* MFCC

## 2. MFCC Extraction

[Notebook](./2-mfcc-extractor.ipynb)

<p align="left">
<img width="500" src="./img/wave_to_mfcc.jpg">
</p>

Utility to read all the .wav files stored in separate folders according to their genre, extract MFCC, and store these values in json format.

## 3. Genre Classifier using GTZAN Dataset

[Notebook](3-gtzan-neural-network.ipynb)

<p align="left">
<img width="250" src="./img/train_test_plot_LSTM_white_bg.jpg">
</p>

* Neural Network
* Improved Neural Network
* Convolutional Neural Network (CNN)
* RNN - LSTM
