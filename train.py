from main import CALM
from utils import Device, BATCH_SIZE, LEARNING_RATE, EPOCHS
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

def dataset(Text,Audio):
    return Text , Audio

class CALM(nn.Module):
    def __init__(self):
        super(CALM, self).__init__()
        # Define layers here, e.g., encoders, attention, pooling, dense, output layers
        self.encoder_model = self.dummy_encoder
        self.CrossAttention = nn.MultiheadAttention(embed_dim=256, num_heads=8)
        self.self_attention = nn.MultiheadAttention(embed_dim=256, num_heads=8)
        self.transform_encoder = nn.TransformerEncoderLayer(d_model=256, nhead=8)
        self.Pooling_Layer = nn.AdaptiveAvgPool1d(1)
        self.DenseLayer = nn.Linear(256, 128)
        self.Output = nn.Linear(128, 1) 

    def dummy_encoder(self, audio_path, text): 
        audio_enc = torch.randn(1, 256)  # Dummy tensor
        text_enc = torch.randn(1, 256)   # Dummy tensor
        return audio_enc, text_enc

    def forward(self, audio_path, text):
        audio_enc, text_enc = self.encoder_model(audio_path, text)
        query, key, value = audio_enc, text_enc, text_enc
        cross_output, _ = self.CrossAttention(query, key, value)
        selfattn_output, _ = self.self_attention(cross_output, cross_output, cross_output)
        encoding_transformer = self.transform_encoder(selfattn_output)
        pooling_output = self.Pooling_Layer(encoding_transformer)
        pooling_output = pooling_output.view(pooling_output.size(0), -1)  # Flatten
        dense_output = self.DenseLayer(pooling_output)
        output = self.Output(dense_output)
        return output



def train(audio_paths, texts, labels, model, epochs=250, batch_size=16, learning_rate=0.001):
    dataset = AudioTextDataset(audio_paths, texts, labels)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        for i, (audio_path, text, label) in enumerate(dataloader):
            optimizer.zero_grad()

            outputs = model(audio_path, text)
            loss = criterion(outputs, label)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if i % 10 == 9:  # Print every 10 batches
                print(f"[Epoch {epoch + 1}, Batch {i + 1}] loss: {running_loss / 10:.3f}")
                running_loss = 0.0

    print('Finished Training')

# Example usage
audio_paths = ['audio1.wav', 'audio2.wav']  # Replace with actual paths
texts = ['This is a sample text.', 'Another text input.']
labels = torch.tensor([[1.0], [0.0]])  # Replace with actual labels

model = CALM()
train(audio_paths, texts, labels, model, epochs=250)



