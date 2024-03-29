import re

import requests
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials

from src.utils.misc import make_directory

out_path = "./../../data/playlist-tracks/"


def request_playlist(client_id, client_secret, playlist_id):
    """
    Authenticate using Spotify credentials to gain access to Spotify's API
    :param client_id: client ID of the "Spotify for Developers" account
    :param client_secret: Secret key corresponding to the client
    :param playlist_id: ID of the playlist of interest
    :return: Authenticated Spotipy object, and the playlist
    """
    auth_manager = SpotifyClientCredentials(client_id=client_id, client_secret=client_secret)
    sp = spotipy.Spotify(auth_manager=auth_manager)
    playlist = sp.playlist(playlist_id=playlist_id)
    return sp, playlist


def show_tracks(tracks):
    """
    Iterate through the playlist and display each song's basic info
    :param tracks: a dict containing all the tracks' details in a list "items"
    :return:
    """
    for item in tracks['items']:
        track = item['track']
        print("Track: ", track['name'])
        print("Artist: ", track['artists'][0]['name'])
        print("Audio preview: ", track['preview_url'])
        print("Album: ", track['album']['name'])
        print("\n")


def download_track_preview(tracks, playlist_name, out_path):
    """
    Download a 30-second track preview of all the songs in the playlist
    :param tracks: A dict containing all the tracks' details in a list "items"
    :param playlist_name: Name of the playlist for folder name to save tracks into

    :return: path: Path to output mp3 directory
    """
    path = out_path + playlist_name + "/mp3/"
    make_directory(path=path)
    for item in tracks['items']:
        track = item['track']
        track_name = re.sub('[^A-Za-z0-9]+', '', track['name'])
        clip_to_download = track['preview_url']
        if clip_to_download is not None:
            response = requests.get(clip_to_download)
            open(file=path + track_name + ".mp3", mode="wb").write(response.content)
    print(f"{len(tracks)} tracks downloaded.")
    return path


if __name__ == "__main__":
    client_id = ""
    client_secret = ""
    playlist_id = "spotify:playlist:7eYZpOTqL0Y3kwEsxNr0PI"

    sp, playlist = request_playlist(client_id=client_id, client_secret=client_secret, playlist_id=playlist_id)

    print(f"Playlist name: {playlist['name']}")
    print(f"Total tracks: {playlist['tracks']['total']}\n\n")

    results = sp.playlist(playlist['id'], fields="tracks,next")
    tracks = results['tracks']
    show_tracks(tracks)
    download_track_preview(tracks=tracks, playlist_name=playlist['name'], out_path=out_path)
