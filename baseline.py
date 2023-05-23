#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.9
# creator: TAN Zusheng

from parser import args_parser
from model import LSTM, RNN, CNN_GRU, CNN_RNN, GRU
import librosa
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from librosa.feature import melspectrogram
from sklearn.model_selection import train_test_split
from pathlib import Path
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

# extract features from audio
def extract_features(file_path, seq_len):
    '''
    Developed by: TAN Zusheng
    This is the function to extract features from audio data
    '''
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


# word tokenizer
class WordTokenizer:
    '''
    Developed by: TAN Zusheng
    This is the wordtokenizer class to tokenize the text
    '''

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


def text_to_int(text, max_len, word_token):
    '''
    Developed by: TAN Zusheng
    This is the function to transform the text into number
    '''
    encoded_text = word_token.encode(eval(text))
    return encoded_text + [0] * (max_len - len(encoded_text))


# Dataset
class MusicDescriptionDataset(Dataset):
    '''
    Developed by: TAN Zusheng
    This is the dataset class to map the data with aspect list
    '''

    def __init__(self, df, data_dir, seq_len, word_token):
        self.data = df
        self.data_dir = data_dir
        self.seq_len = seq_len
        self.word_token = word_token
        self.max_text_len = self.data['aspect_list'].apply(lambda x: len(eval(x))).max()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        audio_file = os.path.join(self.data_dir, f"{row['ytid']}.wav")
        audio_features = extract_features(audio_file, self.seq_len)
        text = row['aspect_list']
        text_int = text_to_int(text, self.max_text_len, self.word_token)
        return torch.FloatTensor(audio_features), torch.LongTensor(text_int)


def train(model, dataloader, criterion, optimizer, device, output_dim):
    '''
    Developed by: TAN Zusheng
    This is the training function for the model training and training loss
    '''
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
    '''
    Developed by: TAN Zusheng
    This is the evaluation function for the model evaluation loss
    '''
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


def test(model, dataloader, criterion, device):
    '''
    Developed by: TAN Zusheng
    This is the test function for the model testing loss
    '''
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


def predict(model, audio_file, seq_len, device, max_length, word_token):
    '''
    Developed by: TAN Zusheng
    This is the predict function for RNN and GRU model
    Using the trained RNN and GRU model to predict and get the decoded music tags (text)
    '''
    model.eval()
    audio_features = extract_features(audio_file, seq_len)
    audio_features = torch.FloatTensor(audio_features).unsqueeze(0).to(device)

    with torch.no_grad():
        _, hidden = model.encoder(audio_features)
        hidden = hidden.expand(model.decoder.num_layers, -1, -1)
        output_tokens = []
        input_token = torch.tensor([[0]]).to(device)  # Start with the first token (e.g., 0)
        for _ in range(max_length):
            tgt_embedded = model.embedding(input_token)
            output, hidden = model.decoder(tgt_embedded, hidden)
            output_token = output.argmax(dim=-1).item()
            output_tokens.append(output_token)
            input_token = torch.tensor([[output_token]]).to(device)

    return ', '.join(word_token.decode(output_tokens))


# LSTM Predict
def lstm_predict(model, audio_file, seq_len, device, max_length, word_token):
    '''
    Developed by: TAN Zusheng
    This is the predict function for CNN_LSTM model
    Using the trained CNN_LSTM model to predict and get the decoded music tags (text)
    '''
    model.eval()
    audio_features = extract_features(audio_file, seq_len)
    audio_features = torch.FloatTensor(audio_features).unsqueeze(0).to(device)

    with torch.no_grad():
        _, (hidden, cell) = model.encoder(audio_features)
        hidden = hidden.expand(model.decoder.num_layers, -1, -1)
        output_tokens = []
        input_token = torch.tensor([[0]]).to(device)  # Start with the first token (e.g., 0)
        for _ in range(max_length):
            tgt_embedded = model.embedding(input_token)
            output, (hidden, cell) = model.decoder(tgt_embedded, (hidden, cell))
            output_token = output.argmax(dim=-1).item()
            output_tokens.append(output_token)
            input_token = torch.tensor([[output_token]]).to(device)

    return ', '.join(word_token.decode(output_tokens))


# CNN_GRU Prediting
def cnn_gru_predict(model, audio_file, seq_len, device, max_length, word_token):
    '''
    Developed by: TAN Zusheng
    This is the predict function for CNN_GRU model
    Using the trained CNN_GRU model to predict and get the decoded music tags (text)
    '''
    model.eval()
    audio_features = extract_features(audio_file, seq_len)
    audio_features = torch.FloatTensor(audio_features).unsqueeze(0).to(device)

    with torch.no_grad():
        # Pass through the encoder
        audio_features = audio_features.permute(0, 2, 1)  # Rearrange input dimensions to fit Conv1D
        conv_out = F.relu(model.encoder_cnn(audio_features))
        conv_out = conv_out.permute(0, 2, 1)  # Rearrange back to fit LSTM
        _, hidden = model.encoder_rnn(conv_out)
        hidden = hidden.expand(model.decoder.num_layers, -1, -1)
        output_tokens = []
        input_token = torch.tensor([[0]]).to(device)  # Start with the first token (e.g., 0)
        for _ in range(max_length):
            tgt_embedded = model.embedding(input_token)
            output, hidden = model.decoder(tgt_embedded, hidden)
            output_token = output.argmax(dim=-1).item()
            output_tokens.append(output_token)
            input_token = torch.tensor([[output_token]]).to(device)

    return ', '.join(word_token.decode(output_tokens))


def main():
    '''
    Developed by: TAN Zusheng
    This is the main function for baseline models.
    Audio data procession --> Model Construction --> Model Training, Testing and Inference
    '''

    ### Define args_parser ###
    args = args_parser()

    ### Training_data path ###
    DATA_DIR = args.data_path

    ### Model parameters ###
    num_layers = 1
    hidden_size = 64
    dropout_rate = 0.2
    BATCH_SIZE = 16
    SEQ_LEN = 512
    NUM_EPOCHS = args.epochs
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ### Read table ###
    descriptions_df = pd.read_csv("music_text_table.csv")

    existing_ids = {f.stem for f in Path(DATA_DIR).glob("*.wav")}
    df = descriptions_df[descriptions_df["ytid"].isin(existing_ids)].reset_index()[["ytid","aspect_list"]]
    
    ### Create a vocabulary from the entire 'aspect_list' column ###
    unique_words = set()
    for aspect_list in df['aspect_list']:
        words = eval(aspect_list)
        unique_words.update(words)

    word_tokenizer = WordTokenizer()
    for word in unique_words:
        word_tokenizer.vocab[word] = len(word_tokenizer.vocab)
    word_tokenizer.inv_vocab = {v: k for k, v in word_tokenizer.vocab.items()}

    ### DataLoader ###
    data = MusicDescriptionDataset(df, DATA_DIR, SEQ_LEN, word_tokenizer)
    train_data, val_data = train_test_split(data, test_size=0.3, random_state=42)
    val_data, test_data = train_test_split(val_data, test_size=0.5, random_state=42)
    
    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(val_data, batch_size=BATCH_SIZE, shuffle=False)
    
    ### Define Model ###
    input_dim = train_loader.dataset[0][0].shape[1]
    output_dim = len(word_tokenizer.vocab)
    
    if args.model == "lstm":
        model = LSTM(input_dim, hidden_size, output_dim, num_layers, dropout_rate).to(DEVICE)
    elif args.model == "gru":
        model = GRU(input_dim, hidden_size, output_dim, num_layers, dropout_rate).to(DEVICE)
    elif args.model == "cnn_rnn":
        model = CNN_RNN(input_dim, hidden_size, output_dim, num_layers, dropout_rate).to(DEVICE)       
    elif args.model == "cnn_gru":
        model = CNN_GRU(input_dim, hidden_size, output_dim, num_layers, dropout_rate).to(DEVICE)    
    elif args.model == "rnn":
        model = RNN(input_dim, hidden_size, output_dim, num_layers, dropout_rate).to(DEVICE)
                
    ### Training setting ###
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    #scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=5, verbose=True)
    
    ### Training ###
    if args.mode == "train":
        best_val_loss = float('inf')
        for epoch in range(NUM_EPOCHS):
            train_loss = train(model, train_loader, criterion, optimizer, DEVICE, output_dim)
            val_loss = evaluate(model, val_loader, criterion, DEVICE)
            print(f"Epoch {epoch + 1}/{NUM_EPOCHS}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
            #scheduler.step(val_loss)  # Add this line

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(model.state_dict(), f'base_{args.model}_model.pt')
        print(f"Training completed, the testing loss is {test(model, test_loader, criterion, DEVICE)}")

    ### Inference ###
    else:
        # Define model structure
        if args.model == "rnn":
            best_model = RNN(input_dim, hidden_size, output_dim, num_layers, dropout_rate).to(DEVICE)
        elif args.model == "lstm":
            best_model = LSTM(input_dim, hidden_size, output_dim, num_layers, dropout_rate).to(DEVICE)
        elif args.model == "gru":
            best_model = GRU(input_dim, hidden_size, output_dim, num_layers, dropout_rate).to(DEVICE)
        elif args.model == "cnn_rnn":
            best_model = CNN_RNN(input_dim, hidden_size, output_dim, num_layers, dropout_rate).to(DEVICE)        
        elif args.model == "cnn_gru":
            best_model = CNN_GRU(input_dim, hidden_size, output_dim, num_layers, dropout_rate).to(DEVICE)        


        best_model.load_state_dict(torch.load(f'base_{args.model}_model.pt'))
        MAX_OUTPUT_LENGTH = 5  # Maximum number of words to output

        # Predict the text description for a new .wav file
        audio_file = "test/ZwLfj7tvpdc.wav"

        # use different pred function to get the decoded text
        if args.model == "cnn_gru":
            predicted_text = cnn_gru_predict(best_model, audio_file, SEQ_LEN, DEVICE, MAX_OUTPUT_LENGTH, word_tokenizer)
        elif args.model == "cnn_rnn":
            predicted_text = cnn_gru_predict(best_model, audio_file, SEQ_LEN, DEVICE, MAX_OUTPUT_LENGTH, word_tokenizer)
        elif args.model == "lstm":
            predicted_text = lstm_predict(best_model, audio_file, SEQ_LEN, DEVICE, MAX_OUTPUT_LENGTH, word_tokenizer)
        else:
            predicted_text = predict(best_model, audio_file, SEQ_LEN, DEVICE, MAX_OUTPUT_LENGTH, word_tokenizer)
        print(f"Predicted music tags are: {predicted_text}")
        
        
if __name__ == "__main__":
    main()
