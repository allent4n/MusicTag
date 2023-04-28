import os
import pandas as pd
import librosa
import numpy as np
import torch
from torch.utils.data import Dataset
from nltk.tokenize import word_tokenize
import os
import librosa
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader
import os
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from librosa.feature import melspectrogram
from sklearn.model_selection import train_test_split
from librosa import load
from pathlib import Path
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

# Constants
DATA_DIR = "1k_data"
BATCH_SIZE = 32
SEQ_LEN = 100
NUM_EPOCHS = 50
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#读取数据table
descriptions_df = pd.read_csv("music_text_table.csv")
existing_ids = {f.stem for f in Path(DATA_DIR).glob("*.wav")}
df = descriptions_df[descriptions_df["ytid"].isin(existing_ids)].reset_index()[["ytid","aspect_list"]]

# Preprocessing
def extract_features(file_path, seq_len=SEQ_LEN):
    y, sr = librosa.load(file_path)
    mfcc = librosa.feature.mfcc(y=y, sr=sr)
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    features = np.vstack((mfcc, chroma))
    if features.shape[1] < seq_len:
        padding = np.zeros((features.shape[0], seq_len - features.shape[1]))
        features = np.hstack((features, padding))
    else:
        features = features[:, :seq_len]
    return features.T

#创建词向量
class WordTokenizer:
    def __init__(self, vocab=None):
        if vocab is None:
            self.vocab = {'<unk>': 0}
        else:
            self.vocab = vocab
        self.inv_vocab = {v: k for k, v in self.vocab.items()}

    def encode(self, words):
        return [self.vocab.get(word, self.vocab['<unk>']) for word in words]

    def decode(self, encoded_words):
        return [self.inv_vocab.get(word_idx, '') for word_idx in encoded_words]

# Create a vocabulary from the entire 'aspect_list' column
unique_words = set()
for aspect_list in df['aspect_list']:
    words = eval(aspect_list)
    unique_words.update(words)

word_tokenizer = WordTokenizer()
for word in unique_words:
    word_tokenizer.vocab[word] = len(word_tokenizer.vocab)
word_tokenizer.inv_vocab = {v: k for k, v in word_tokenizer.vocab.items()}

def text_to_int(text, max_len):
    encoded_text = word_tokenizer.encode(eval(text))
    return encoded_text + [0] * (max_len - len(encoded_text))

# Dataset
class MusicDescriptionDataset(Dataset):
    def __init__(self, df, data_dir, seq_len=SEQ_LEN):
        self.data = df
        self.data_dir = data_dir
        self.seq_len = seq_len
        self.max_text_len = self.data['aspect_list'].apply(lambda x: len(eval(x))).max()
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        audio_file = os.path.join(self.data_dir, f"{row['ytid']}.wav")
        audio_features = extract_features(audio_file, self.seq_len)
        text = row['aspect_list']
        text_int = text_to_int(text, self.max_text_len)
        return torch.FloatTensor(audio_features), torch.LongTensor(text_int)

# DataLoader
data = MusicDescriptionDataset(df, DATA_DIR)
train_data, val_data = train_test_split(data, test_size=0.2, random_state=42)
train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_data, batch_size=BATCH_SIZE, shuffle=False)

# model
class Seq2SeqModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(Seq2SeqModel, self).__init__()
        self.encoder = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.decoder = nn.LSTM(hidden_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
        self.embedding = nn.Embedding(output_size, hidden_size)  # Add this line
    def forward(self, src, tgt):
        _, (hidden, cell) = self.encoder(src)
        tgt_embedded = self.embedding(tgt)  # Add this line
        outputs, _ = self.decoder(tgt_embedded, (hidden, cell))
        return self.fc(outputs)


input_dim = train_loader.dataset[0][0].shape[1]
output_dim = len(word_tokenizer.vocab)  # Adjust this based on the range of characters you expect
model = Seq2SeqModel(input_dim, output_dim, output_dim).to(DEVICE)
    
# Training & Evaluation
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)
def train(model, dataloader, criterion, optimizer, device):
    model.train()
    epoch_loss = 0

    for batch in dataloader:
        audio_features, text_int = batch
        audio_features = audio_features.to(device)
        text_int = text_int.to(device)
        tgt_input = text_int[:, :-1].long()
        # print(audio_features.shape,text_int.shape,tgt_input.shape)
        
        assert tgt_input.min() >= 0 and tgt_input.max() < output_dim, f"Invalid target input indices: min={tgt_input.min()}, max={tgt_input.max()}"

        optimizer.zero_grad()
        output = model(audio_features, tgt_input)

        output = output.contiguous().view(-1, output.shape[-1])
        text_int = text_int[:, 1:].contiguous().view(-1)

        loss = criterion(output, text_int)
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

    return epoch_loss / len(dataloader)

def evaluate(model, dataloader, criterion, device):
    model.eval()
    epoch_loss = 0

    with torch.no_grad():
        for batch in dataloader:
            audio_features, text_int = batch
            audio_features = audio_features.to(device)
            text_int = text_int.to(device)

            tgt_input = text_int[:, :-1].long()
            output = model(audio_features, tgt_input)

            output = output.contiguous().view(-1, output.shape[-1])
            text_int = text_int[:, 1:].contiguous().view(-1)

            loss = criterion(output, text_int)
            epoch_loss += loss.item()

    return epoch_loss / len(dataloader)

best_val_loss = float('inf')
for epoch in range(NUM_EPOCHS):
    train_loss = train(model, train_loader, criterion, optimizer, DEVICE)
    val_loss = evaluate(model, val_loader, criterion, DEVICE)
    print(f"Epoch {epoch + 1}/{NUM_EPOCHS}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), 'best_model.pt')
print("Training complete.")


def predict(model, audio_file, seq_len, device):
    model.eval()
    audio_features = extract_features(audio_file, seq_len)
    audio_features = torch.FloatTensor(audio_features).unsqueeze(0).to(device)

    with torch.no_grad():
        _, (hidden, cell) = model.encoder(audio_features)
        hidden, cell = hidden.expand(model.decoder.num_layers, -1, -1), cell.expand(model.decoder.num_layers, -1, -1)
        output_tokens = []
        input_token = torch.tensor([[0]]).to(device)  # Start with the first token (e.g., 0)
        for _ in range(MAX_OUTPUT_LENGTH):
            tgt_embedded = model.embedding(input_token)
            output, (hidden, cell) = model.decoder(tgt_embedded, (hidden, cell))
            output_token = output.argmax(dim=-1).item()
            output_tokens.append(output_token)
            input_token = torch.tensor([[output_token]]).to(device)

    return ', '.join(word_tokenizer.decode(output_tokens))


# Load the best model
best_model = Seq2SeqModel(input_dim, output_dim, output_dim).to(DEVICE)
best_model.load_state_dict(torch.load('best_model.pt'))
MAX_OUTPUT_LENGTH = 5  #要输出的最大词数量
# Predict the text description for a new .wav file
# audio_file = DATA_DIR + '/_6C2ffY_-mc.wav'   

audio_file = "test/ZwLfj7tvpdc.wav"
predicted_text = predict(best_model, audio_file, SEQ_LEN, DEVICE)
print(f"Predicted text: {predicted_text}")






