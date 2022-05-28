import os
import subprocess

from src.utils import make_directory

input_dir = "./../data/playlist-tracks/20th Century/"
output_dir = "./../data/playlist-tracks/mp3-to-wav/"
path_to_ffmpeg_exe = "./../extras/ffmpeg-2022-05-26-git-0dcbe1c1aa-full_build/bin/ffmpeg.exe"


def convert_mp3_to_wav(input_dir, output_dir,
                       ffmpeg_path="./../extras/ffmpeg-2022-05-26-git-0dcbe1c1aa-full_build/bin/ffmpeg.exe"):
    """
    - Convert all the mp3 files in the input_dir to wav, and save them to output_dir
    - Will overwrite if the wav file of the same name already exists in the output directory
    :param input_dir: Path to input directory
    :param output_dir: Path to output directory
    :param ffmpeg_path: Path to the ffmpeg.exe file (required for handling mp3 files)
    :return:
    """
    make_directory(path=output_dir)
    for i, (dirpath, dirnames, filenames) in enumerate(os.walk(input_dir)):
        for file in filenames:
            print(file)
            subprocess.call([ffmpeg_path, '-i', os.path.join(input_dir, file), '-y',
                             os.path.join(output_dir, str(file.split(".")[0] + ".wav"))])


if __name__ == "__main__":
    convert_mp3_to_wav(input_dir=input_dir, output_dir=output_dir)
