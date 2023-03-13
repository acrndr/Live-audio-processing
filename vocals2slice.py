import os
import auditok
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="transcribe audio to text")
    parser.add_argument("-a", "--audio_dir", type=str, required=True, help="audio file(s) to transcribe")
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
    
    length = 0
    for file in output_dir:
        if file[-3:] == "wav":
            if int(file[:-4]) > length:
                length = int(file[:-4])
    length += 1

    for file in os.listdir(audio_dir):
        if file[-3:] != "wav":
            continue
        print("start to separate " + file)
        audio_regions = auditok.split(
                os.path.join(audio_dir, file),
                min_dur=1.3,  # minimum duration of a valid audio event in seconds
                max_dur=10,  # maximum duration of an event
                max_silence=0.5,  # maximum duration of tolerated continuous silence within an event
                energy_threshold=55  # threshold of detectiony
        )
        before = length
        for i, r in enumerate(audio_regions):
            r.save(os.path.join(output_dir, "{:06d}".format(length) + ".wav"))
            length += 1
        print(file + " are separated to " + str(length - before) + " wavs")
    print("done")