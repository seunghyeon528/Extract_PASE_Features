#-*-coding:utf-8-*-
### collate 사용해서 batch 단위로 빠르게 inference 하기 ###

import torch
import pase
import pdb
from pase.models.frontend import wf_builder
import os
import csv
import glob
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset
import librosa
from tqdm import tqdm
import numpy as np
import math

########################################################
##                     PASE Feature
########################################################

# 1. GPU configuration
os.environ["CUDA_VISIBLE_DEVICES"]="0"
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(device)

 
# 2. Prepare list of audio_path
audio_list = [] 
csv_path = "./csv" # symbolic link
csv_list = glob.glob(os.path.join(csv_path, "*.csv"))
pdb.set_trace()
for csv_file in csv_list:
    with open(csv_file, newline='',encoding = "utf-8") as f:
        reader = csv.reader(f)
        tmp = list(reader)
        temp_list = [x[0] for x in tmp[1:]]
        audio_list.extend(temp_list)


# 3. Data_Reader
class Data_Reader(Dataset):
    def __init__(self, audio_list):
        self.audio_list = audio_list
    
    def __len__(self):
        return len(self.audio_list)
    
    def get_savepath(self,original_path):
        # pdb.set_trace
        root_dir = "./save_root" # symbolic link for root directory of saving path
        full_path = os.path.join(root_dir, original_path)
        
        return str(full_path)

    def __getitem__(self,idx):
        save_path = self.get_savepath(str(self.audio_list[idx]))
        audio_path = "./audio_root/"+  str(self.audio_list[idx])
        audio, _ = librosa.load(audio_path, sr=16000, dtype=np.float32)
        audio = torch.FloatTensor(audio)
        audio = audio.unsqueeze(0)
        return audio, save_path

dataset = Data_Reader(audio_list)


# 4. DataLoader
def my_collate(batch):
    audios,save_paths = list(zip(*batch))
    # find max length of audios in batch
    audios_len = torch.LongTensor(np.array([x.shape[1] for x in audios]))
    max_len = max(audios_len)

    # zero padding
    batch_size = len(audios)
    inputs = torch.zeros(batch_size,1,max_len)
    for i in range(batch_size):
        inputs[i,:, :audios[i].shape[1]] = audios[i]

    return (inputs, save_paths, audios_len)
    

loader = DataLoader(dataset = dataset, batch_size=8, shuffle=False, collate_fn=lambda x: my_collate(x))
print(len(loader))


# 5. Model
pase = wf_builder('cfg/frontend/PASE+.cfg').eval()
pase.load_pretrained('FE_e199.ckpt', load_last=True, verbose=True)
pase = pase.cuda()


# 6. Inference
with torch.no_grad():
    for i, (inputs, save_paths, audios_len) in enumerate(tqdm(loader)):
        inputs = inputs.to(device) # GPU load
        output_feature = pase(inputs.cuda(),device = "cuda")
        output_feature = output_feature.detach().cpu().numpy()
        # save features
        for i in range(len(audios_len)):
            audio_len = audios_len[i]
            feature_len = math.floor(audio_len/160)
            pase_feature = (output_feature[i])[:,:feature_len]
            save_path = save_paths[i]
            save_path = save_path.replace(".wav",".npy")
            dir_path = os.path.dirname(save_path)
            if not os.path.exists(dir_path):
                os.makedirs(dir_path)
            np.save(save_path,pase_feature)

            
