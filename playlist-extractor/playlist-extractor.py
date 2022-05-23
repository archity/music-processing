import spotipy
from spotipy.oauth2 import SpotifyClientCredentials


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
