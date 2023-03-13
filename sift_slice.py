import argparse, os, torch, warnings
import numpy as np
import torch
from torch.nn import functional as F
from model.ECAPA_TDNN import ECAPA_TDNN
import soundfile
from tqdm import tqdm
warnings.simplefilter("ignore")


def load_model(dict_path, device):
    encoder = ECAPA_TDNN(C=1024)
    state_dict = torch.load(dict_path)
    encoder_state = encoder.state_dict()
    for name, param in state_dict.items():
        module = name.split(".")[0]
        true_name = name[len(module) + 1:]
        if module == "speaker_encoder":
            encoder_state[true_name].copy_(param)
    encoder.to(device)
    encoder.eval()
    return encoder


def get_embedding(path, encoder):
    embeddings = []
    device = next(encoder.parameters()).device
    for file in tqdm(path):
        audio, _  = soundfile.read(file)
        if len(audio.shape) > 1:
            audio = audio[:, 0]
        data_1 = torch.tensor(np.stack([audio],axis=0), device=device, dtype=torch.float32)
        max_audio = 300 * 160 + 240
        if audio.shape[0] <= max_audio:
            shortage = max_audio - audio.shape[0]
            audio = np.pad(audio, (0, shortage), 'wrap')
        feats = []
        startframe = np.linspace(0, audio.shape[0] - max_audio, num=5)
        for asf in startframe:
            feats.append(audio[int(asf):int(asf) + max_audio])
        feats = np.stack(feats, axis=0).astype(np.float)
        data_2 = torch.tensor(feats, device=device, dtype=torch.float32)
        with torch.no_grad():
            embedding_1 = encoder.forward(data_1, aug=False)
            embedding_1 = F.normalize(embedding_1, p=2, dim=1)
            embedding_2 = encoder.forward(data_2, aug=False)
            embedding_2 = F.normalize(embedding_2, p=2, dim=1)
        embeddings.append([embedding_1, embedding_2])
    return embeddings


def get_score(embedding0, embedding1):
    embedding_11, embedding_12 = embedding0
    score_list = []
    for embedding in embedding1:
        embedding_21, embedding_22 = embedding
        score_1 = torch.mean(torch.matmul(embedding_11, embedding_21.T))
        score_2 = torch.mean(torch.matmul(embedding_12, embedding_22.T))
        score = (score_1 + score_2) / 2
        score_list.append(score.detach().cpu())
    return score_list


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="select similar voice")
    parser.add_argument("-d", "--dict", type=str, default="model/ECAPA_TDNN/pretrain.model", help="path of state dictionary")
    parser.add_argument("-c", "--contrast", type=str, required=True, help="path of contrast audio")
    parser.add_argument("-f", "--file_dir", type=str, required=True, help="path of file dir need to select")
    parser.add_argument("-t", "--threshold", type=float, default=0.3, help="threshold for the accepted audio score, it should be 0~1")
    parser.add_argument("--device", type=str, default="", help="device of model")
    # parser.add_argument("-o", "--output_dir", default="./", help="path to save output file")
    args = parser.parse_args()

    dict_path = args.dict
    if not os.path.isfile(dict_path):
        print("dictionary " + dict_path + " is not exist!")
        exit()
    
    contrast_path = args.contrast
    if not os.path.isfile(contrast_path):
        print("contrast audio " + contrast_path + " is not exist!")
        exit()

    eval_path = args.file_dir
    if not os.path.isdir(eval_path):
        print("file directory " + eval_path + " is not exist!")
        exit()

    threshold = args.threshold
    if threshold < 0 or threshold > 1:
        print("got threshold " + str(threshold) + ", the threshold should be between 0~1")
        exit()

    device = args.device
    if device == "":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # output_dir = args.output_dir
    # if not os.path.isdir(output_dir):
    #     print("output directory " + output_dir + " is not exist!")
    
    print("load model...")
    encoder = load_model(dict_path, device)
    print("model loading complete!")

    print("load contrast audio...")
    contrast_embedding = get_embedding([contrast_path], encoder)[0]
    print("contrast audio loading complete!")

    file_list = []
    for file in os.listdir(eval_path):
        if file[-3:] == "wav":
            file_list.append(os.path.join(eval_path, file))
    
    print("load verification audio...")
    eval_embedding = get_embedding(file_list, encoder)
    print("verification audio loading complete!")

    print("calculating similarity scores...")
    score_list = get_score(contrast_embedding, eval_embedding)
    print("finish")

    print("output similar audio list")
    not_similar_audio = []
    similar_audio = []
    for score, file in zip(score_list, file_list):
        if score < threshold:
            not_similar_audio.append(file)
        else:
            similar_audio.append(file)
        
    os.makedirs(os.path.join(eval_path, "dirty"), exist_ok=True)
    for file in not_similar_audio:
        os.rename(os.path.join(eval_path, file), os.path.join(eval_path, "dirty", file))
    
    similar_audio.sort()
    length = 0
    for file in similar_audio:
        os.rename(os.path.join(eval_path, file), os.path.join(eval_path, "{:06d}".format(length) + ".wav"))
        length += 1

    # with open(os.path.join(output_dir, "not_similar_audio.txt"), "w") as f:        
    #     f.writelines(not_similar_audio)
    # with open(os.path.join(output_dir, "similar_audio.txt"), "w") as f:        
    #     f.writelines(similar_audio)

    print("done")     