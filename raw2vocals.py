import os
import subprocess
import argparse
from demucs.apply import apply_model
from demucs.audio import AudioFile, convert_audio, save_audio
from demucs.pretrained import get_model
import torchaudio as ta
import torch

def load_track(track, audio_channels, samplerate):
    errors = {}
    wav = None

    try:
        wav = AudioFile(track).read(
            streams=0,
            samplerate=samplerate,
            channels=audio_channels)
    except FileNotFoundError:
        errors['ffmpeg'] = 'Ffmpeg is not installed.'
    except subprocess.CalledProcessError:
        errors['ffmpeg'] = 'FFmpeg could not read the file.'

    if wav is None:
        try:
            wav, sr = ta.load(str(track))
        except RuntimeError as err:
            errors['torchaudio'] = err.args[0]
        else:
            wav = convert_audio(wav, sr, samplerate, audio_channels)

    if wav is None:
        print(f"Could not load file {track}. "
              "Maybe it is not a supported file format? ")
        for backend, error in errors.items():
            print(f"When trying to load using {backend}, got the following error: {error}")
        exit(1)
    return wav

def separate_and_save(model, file_list, device):
    for file, save_file in file_list:
        print("start to spearate " + file)
        wav = load_track(file, model.audio_channels, model.samplerate)
        ref = wav.mean(0)
        wav = (wav - ref.mean()) / ref.std()
        print(file + " load success")
        sources = apply_model(model, wav[None], device=device)[0][-1]
        sources = sources * ref.std() + ref.mean()
        print("start to save " + save_file)
        save_audio(sources, save_file, model.samplerate)
        print(save_file + "wav have been save")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="transcribe audio to text")
    parser.add_argument("-a", "--audio_dir", type=str, required=True, help="audio file(s) to transcribe")
    parser.add_argument("-d", "--device", type=str, default=None, help="device to use for PyTorch inference")
    parser.add_argument("-o", "--output_dir", type=str, default=".", help="directory to save the outputs")
    args = parser.parse_args()
    audio_dir = args.audio_dir
    if not os.path.isdir(audio_dir):
        print(audio_dir + " is not a directory")
        exit(1)

    output_dir = args.output_dir
    if not os.path.isdir(output_dir):
        print(output_dir + " is not a directory")
        exit(1)

    exist_file = []
    for file in os.listdir(output_dir):
        if file[-3:] == "wav":
            exist_file.append(file[:-3] + "mp3")

    not_sep = []
    for file in os.listdir(audio_dir):
        if file not in exist_file and file[-3:] == "mp3":
            not_sep.append([os.path.join(audio_dir, file), os.path.join(output_dir, file[:-3] + "wav")])

    print("loading model...")
    device = args.device
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    model = get_model("mdx_extra_q", None)
    model.eval().to(device)
    print("model loading complete!")
    print("start to separate")
    separate_and_save(model, not_sep, device)
    print("done")

    