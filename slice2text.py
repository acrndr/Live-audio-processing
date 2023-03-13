from whisper import load_model, transcribe
import argparse
import os
import zhconv
import re
import torch.multiprocessing as mp
from tqdm import tqdm

replace_list = ["、", "`", "，", "。", "《", "》", "？", "！", ",", "."]
ctx = mp.get_context("spawn")

class multi_trans():
    def __init__(self, file_list, device, id) -> None:
        self.file_list = file_list
        self.device = device
        self.id = id
        self.lock = ctx.Lock()

    def trans(self, device, audio_list, id):
        print("load model in " + device)
        model = load_model("medium", device=device)
        zhmodel = re.compile(u"^[\u4E00-\u9FFF\s]+$")
        for audio in audio_list:
            result = transcribe(model, audio, fp16=True)
            text = result["text"]
            text = zhconv.convert(text, 'zh-cn')
            global replace_list
            for char in replace_list:
                text = text.replace(char, " ")
            if zhmodel.match(text):
                self.lock.acquire()
                with open(str(id) + ".txt", "a") as f:
                    f.write(audio + "|" + str(id) + "|" + text + "\n")
                self.lock.release()
            print(audio + " already transcribed")

    def start(self):
        num_devices = len(self.device)
        process_list = []
        for i, d in enumerate(self.device):
            process = ctx.Process(target=self.trans, args=(d, self.file_list[i::num_devices], self.id))
            process.start()
            process_list.append(process)
        
        for process in process_list:
            if process.is_alive():
                process.join()
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="transcribe audio to text")
    parser.add_argument("-a", "--audio_dir", type=str, help="audio file(s) to transcribe")
    parser.add_argument("-d", "--device", type=str, default=["cuda"], nargs="+", help="device to use for PyTorch inference")
    parser.add_argument("-i", "--id", type=int, default=0, help="the speaker id")
    args = parser.parse_args().__dict__
    
    audio_dir = args.pop("audio_dir")
    if not os.path.isdir(audio_dir):
        print(audio_dir + " is not a directory")
        exit()
    
    device = args.pop("device")
    num_devices = len(device)
    id = args.pop("id")
    
    process_pool = ctx.Pool(num_devices)
    print("start to get file list...")

    with open(str(id) + ".txt", "r") as f:
        lines = f.readlines()
    exist_list = [line.split("|")[0].split("/")[-1] for line in lines]

    file_list = []
    for file in tqdm(os.listdir(audio_dir)):
        if file[-3:] != "wav":
            continue
        if file in exist_list:
            continue
        file_list.append(os.path.join(audio_dir, file))
    print("get file list")

    print("start to transcribe...")
    trans = multi_trans(file_list, device, id)
    trans.start()
    print("done")