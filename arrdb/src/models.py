"""
Deep learning model architectures for Arrhythmia Classification.

Lightweight 1D-CNN and LSTM models for sequence-based classification.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class LightweightCNN1D(nn.Module):
    """
    Lightweight 1D-CNN for arrhythmia sequence classification.
    
    Architecture:
    - Conv1D(1 -> 32, kernel=5)
    - MaxPool1D(2)
    - Conv1D(32 -> 64, kernel=5)
    - MaxPool1D(2)
    - FC(64 * (input_size//4) -> 128)
    - Dropout(0.3)
    - FC(128 -> num_classes)
    """
    
    def __init__(self, input_size=60, num_classes=11):
        super(LightweightCNN1D, self).__init__()
        
        # Convolutional layers
        self.conv1 = nn.Conv1d(1, 32, kernel_size=5, padding=2)
        self.pool1 = nn.MaxPool1d(2)
        
        self.conv2 = nn.Conv1d(32, 64, kernel_size=5, padding=2)
        self.pool2 = nn.MaxPool1d(2)
        
        # Calculate flattened size
        flattened_size = 64 * (input_size // 4)
        
        # Fully connected layers
        self.fc1 = nn.Linear(flattened_size, 128)
        self.dropout = nn.Dropout(0.3)
        self.fc2 = nn.Linear(128, num_classes)
        
    def forward(self, x):
        # x shape: (batch_size, 1, seq_length)
        
        # First conv block
        x = F.relu(self.conv1(x))
        x = self.pool1(x)
        
        # Second conv block
        x = F.relu(self.conv2(x))
        x = self.pool2(x)
        
        # Flatten
        x = x.view(x.size(0), -1)
        
        # Fully connected
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x


class LightweightLSTM(nn.Module):
    """
    Lightweight LSTM for arrhythmia sequence classification.
    
    Architecture:
    - LSTM(input_size=1, hidden_size=64, num_layers=2)
    - Dropout(0.2)
    - FC(64 -> num_classes)
    """
    
    def __init__(self, input_size=1, hidden_size=64, num_layers=2, num_classes=11, dropout=0.2):
        super(LightweightLSTM, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # LSTM layer
        self.lstm = nn.LSTM(
            input_size, 
            hidden_size, 
            num_layers, 
            batch_first=True, 
            dropout=dropout
        )
        
        # Fully connected layer
        self.fc = nn.Linear(hidden_size, num_classes)
        
    def forward(self, x):
        # x shape: (batch_size, seq_length) -> (batch_size, seq_length, 1)
        if x.dim() == 2:
            x = x.unsqueeze(-1)
        
        # LSTM forward
        # output: (batch_size, seq_length, hidden_size)
        # hidden: (h_n, c_n) where h_n: (num_layers, batch_size, hidden_size)
        _, (h_n, _) = self.lstm(x)
        
        # Use the last hidden state from the last layer
        # h_n shape: (num_layers, batch_size, hidden_size)
        last_hidden = h_n[-1]  # (batch_size, hidden_size)
        
        # Fully connected
        out = self.fc(last_hidden)
        
        return out


class LightweightBiLSTM(nn.Module):
    """
    Lightweight Bidirectional LSTM for arrhythmia sequence classification.
    
    Architecture:
    - BiLSTM(input_size=1, hidden_size=64, num_layers=2)
    - Dropout(0.2)
    - FC(128 -> num_classes)  # 128 = 64*2 for bidirectional
    """
    
    def __init__(self, input_size=1, hidden_size=64, num_layers=2, num_classes=11, dropout=0.2):
        super(LightweightBiLSTM, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # Bidirectional LSTM
        self.lstm = nn.LSTM(
            input_size,
            hidden_size,
            num_layers,
            batch_first=True,
            dropout=dropout,
            bidirectional=True
        )
        
        # Fully connected layer (hidden_size*2 for bidirectional)
        self.fc = nn.Linear(hidden_size * 2, num_classes)
        
    def forward(self, x):
        # x shape: (batch_size, seq_length) -> (batch_size, seq_length, 1)
        if x.dim() == 2:
            x = x.unsqueeze(-1)
        
        # LSTM forward
        _, (h_n, _) = self.lstm(x)
        
        # Combine forward and backward hidden states from last layer
        # h_n shape: (num_layers*2, batch_size, hidden_size)
        forward_hidden = h_n[self.num_layers - 1]  # Last forward layer
        backward_hidden = h_n[2 * self.num_layers - 1]  # Last backward layer
        
        # Concatenate forward and backward
        combined_hidden = torch.cat([forward_hidden, backward_hidden], dim=1)
        
        # Fully connected
        out = self.fc(combined_hidden)
        
        return out


class CNNLSTMHybrid(nn.Module):
    """
    Hybrid CNN-LSTM model for arrhythmia classification.
    
    Architecture:
    - Conv1D for local feature extraction
    - LSTM for temporal modeling
    - FC for classification
    """
    
    def __init__(self, input_size=60, num_classes=11, cnn_channels=32, lstm_hidden=64):
        super(CNNLSTMHybrid, self).__init__()
        
        # CNN layers for feature extraction
        self.conv1 = nn.Conv1d(1, cnn_channels, kernel_size=5, padding=2)
        self.pool1 = nn.MaxPool1d(2)
        self.conv2 = nn.Conv1d(cnn_channels, cnn_channels * 2, kernel_size=5, padding=2)
        self.pool2 = nn.MaxPool1d(2)
        
        # LSTM layers
        self.lstm = nn.LSTM(
            cnn_channels * 2,
            lstm_hidden,
            num_layers=1,
            batch_first=True
        )
        
        # Fully connected
        self.fc = nn.Linear(lstm_hidden, num_classes)
        
    def forward(self, x):
        # x shape: (batch_size, 1, seq_length)
        
        # CNN feature extraction
        x = F.relu(self.conv1(x))
        x = self.pool1(x)
        x = F.relu(self.conv2(x))
        x = self.pool2(x)
        
        # Permute for LSTM: (batch, channels, seq) -> (batch, seq, channels)
        x = x.permute(0, 2, 1)
        
        # LSTM
        _, (h_n, _) = self.lstm(x)
        
        # Use last hidden state
        out = self.fc(h_n[-1])
        
        return out

