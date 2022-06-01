from extras.spotify_creds import client_id, client_secret
from src.utils.mfcc_extractor import save_mfcc
from src.utils.mp3_to_wav import convert_mp3_to_wav
from src.utils.playlist_extractor import request_playlist, download_track_preview

out_dir = "./../data/playlist-tracks/"
path_to_ffmpeg_exe = "./../extras/ffmpeg-2022-05-26-git-0dcbe1c1aa-full_build/bin/ffmpeg.exe"
playlist_id = "spotify:playlist:7eYZpOTqL0Y3kwEsxNr0PI"

if __name__ == "__main__":

    # Download playlist tracks
    sp, playlist = request_playlist(client_id=client_id, client_secret=client_secret, playlist_id=playlist_id)
    results = sp.playlist(playlist['id'], fields="tracks,next")
    tracks = results['tracks']
    mp3_dir = download_track_preview(tracks=tracks, playlist_name=playlist['name'], out_path=out_dir)

    # Convert downloaded tracks from mp3 to wav
    wav_dir = convert_mp3_to_wav(input_dir=mp3_dir, ffmpeg_path=path_to_ffmpeg_exe)

    # Extract MFCCs from all the converted wav tracks
    save_mfcc(dataset_path=wav_dir, json_path=wav_dir + "../" + "playlist_songs_mfcc_json.json",
              num_segments=10, subfolders=False)
