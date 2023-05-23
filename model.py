#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.9
# creator: TAN Zusheng

import torch.nn as nn
import torch.nn.functional as F


class LSTM(nn.Module):
    '''
    Developed by: TAN Zusheng
    This is the LSTM-based autoencoder model with one encoder layer and one decoder layer.
    '''
    def __init__(self, input_size, hidden_size, output_size, num_layers, dropout_p=0.1):
        super(LSTM, self).__init__()
        self.encoder = nn.LSTM(input_size, hidden_size, batch_first=True, num_layers=num_layers)
        self.decoder = nn.LSTM(hidden_size, hidden_size, batch_first=True, num_layers=num_layers)
        self.fc = nn.Linear(hidden_size, output_size)
        self.embedding = nn.Embedding(output_size, hidden_size)  # Add this line
    def forward(self, src, tgt):
        _, (hidden, cell) = self.encoder(src)
        tgt_embedded = self.embedding(tgt)  # Add this line
        outputs, _ = self.decoder(tgt_embedded, (hidden, cell))
        return self.fc(outputs)
        

class CNN_LSTM(nn.Module):
    '''
    Developed by: TAN Zusheng
    This is the CNN-LSTM-based autoencoder model with one CNN layer and one LSTM layer as encoder layer and one decoder layer.
    '''
    def __init__(self, input_size, hidden_size, output_size, num_layers, dropout_p=0.1):
        super(CNN_LSTM, self).__init__()
        self.encoder_cnn = nn.Conv1d(input_size, hidden_size, kernel_size=3, padding=1)
        self.encoder_rnn = nn.LSTM(hidden_size, hidden_size, num_layers=num_layers, batch_first=True)
        self.decoder = nn.LSTM(hidden_size, hidden_size, num_layers=num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
        self.embedding = nn.Embedding(output_size, hidden_size)
        self.dropout = nn.Dropout(dropout_p)

    def forward(self, src, tgt):
        src = src.permute(0, 2, 1)  # Rearrange input dimensions to fit Conv1D
        conv_out = self.dropout(F.relu(self.encoder_cnn(src)))
        conv_out = conv_out.permute(0, 2, 1)  # Rearrange back to fit LSTM
        _, (hidden, cell) = self.encoder_rnn(conv_out)
        
        tgt_embedded = self.dropout(self.embedding(tgt))
        outputs, _ = self.decoder(tgt_embedded, (hidden, cell))
        return self.fc(outputs)
        
class CNN_GRU(nn.Module):
    '''
    Developed by: TAN Zusheng
    This is the CNN-GRU-based autoencoder model with one CNN layer and one GRU layer as encoder layer and one decoder layer.
    '''
    def __init__(self, input_size, hidden_size, output_size, num_layers, dropout_p=0.1):
        super(CNN_GRU, self).__init__()
        self.encoder_cnn = nn.Conv1d(input_size, hidden_size, kernel_size=3, padding=1)
        self.encoder_rnn = nn.GRU(hidden_size, hidden_size, num_layers=num_layers, batch_first=True)
        self.decoder = nn.GRU(hidden_size, hidden_size, num_layers=num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
        self.embedding = nn.Embedding(output_size, hidden_size)
        self.dropout = nn.Dropout(dropout_p)

    def forward(self, src, tgt):
        src = src.permute(0, 2, 1)  # Rearrange input dimensions to fit Conv1D
        conv_out = self.dropout(F.relu(self.encoder_cnn(src)))
        conv_out = conv_out.permute(0, 2, 1)  # Rearrange back to fit LSTM
        _, hidden = self.encoder_rnn(conv_out)
        
        tgt_embedded = self.dropout(self.embedding(tgt))
        outputs, _ = self.decoder(tgt_embedded, hidden)
        return self.fc(outputs)
        
class CNN_RNN(nn.Module):
    '''
    Developed by: TAN Zusheng
    This is the CNN-RNN-based autoencoder model with one CNN layer and one RNN layer as encoder layer and one decoder layer.
    '''
    def __init__(self, input_size, hidden_size, output_size, num_layers, dropout_p=0.1):
        super(CNN_RNN, self).__init__()
        self.encoder_cnn = nn.Conv1d(input_size, hidden_size, kernel_size=3, padding=1)
        self.encoder_rnn = nn.RNN(hidden_size, hidden_size, num_layers=num_layers, batch_first=True)
        self.decoder = nn.RNN(hidden_size, hidden_size, num_layers=num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
        self.embedding = nn.Embedding(output_size, hidden_size)
        self.dropout = nn.Dropout(dropout_p)

    def forward(self, src, tgt):
        src = src.permute(0, 2, 1)  # Rearrange input dimensions to fit Conv1D
        conv_out = self.dropout(F.relu(self.encoder_cnn(src)))
        conv_out = conv_out.permute(0, 2, 1)  # Rearrange back to fit LSTM
        _, hidden = self.encoder_rnn(conv_out)
        
        tgt_embedded = self.dropout(self.embedding(tgt))
        outputs, _ = self.decoder(tgt_embedded, hidden)
        return self.fc(outputs)


class GRU(nn.Module):
    '''
    Developed by: TAN Zusheng
    This is the GRU-based autoencoder model with one encoder layer and one decoder layer.
    '''
    def __init__(self, input_size, hidden_size, output_size, num_layers, dropout_p=0.1):
        super(GRU, self).__init__()
        self.encoder = nn.GRU(input_size, hidden_size, num_layers=num_layers, batch_first=True)
        self.decoder = nn.GRU(hidden_size, hidden_size, num_layers=num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
        self.embedding = nn.Embedding(output_size, hidden_size)  # Add this line
    def forward(self, src, tgt):
        _, hidden = self.encoder(src)
        tgt_embedded = self.embedding(tgt)  # Add this line
        outputs, _ = self.decoder(tgt_embedded, hidden)
        return self.fc(outputs)
        
        
class RNN(nn.Module):
    '''
    Developed by: TAN Zusheng
    This is the RNN-based autoencoder model with one encoder layer and one decoder layer.
    '''
    def __init__(self, input_size, hidden_size, output_size, num_layers, dropout_p=0.1):
        super(RNN, self).__init__()
        self.encoder = nn.RNN(input_size, hidden_size, num_layers=num_layers, batch_first=True)
        self.decoder = nn.RNN(hidden_size, hidden_size, num_layers=num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
        self.embedding = nn.Embedding(output_size, hidden_size)  # Add this line
    def forward(self, src, tgt):
        _, hidden = self.encoder(src)
        tgt_embedded = self.embedding(tgt)  # Add this line
        outputs, _ = self.decoder(tgt_embedded, hidden)
        return self.fc(outputs)
