import os
from torch.utils.data import Dataset
import pandas as pd
from torchaudio import load as torchaudio_load

class ClothoAudioTextDataset(Dataset):
    def __init__(self, audio_dir,label_csv, transform=None, target_transform=None):
        self.q_a_df = pd.read_csv(label_csv).dropna().drop(columns=['confidence'])
        self.audio_dir = audio_dir
        self.audio_files = os.listdir(audio_dir)
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.audio_files)    

    def __getitem__(self, idx):
        audio_path = os.path.join(self.audio_dir, self.q_a_df.iloc[idx]['file_name']) 
        question = self.q_a_df.iloc[idx]['QuestionText']
        answer = self.q_a_df.iloc[idx]['answer']
        q_a = '<s>[INST] ' + question + '[/INST]' + answer +'</s>'
        
        if self.transform:
            audio = self.transform(audio)
        if self.target_transform:
            q_a = self.target_transform(q_a)
        return audio_path , q_a

if __name__ == "__main__":
    dataset = ClothoAudioTextDataset('data/clotho-aqa/audio_files','data/clotho-aqa/clotho_aqa_test.csv')
 