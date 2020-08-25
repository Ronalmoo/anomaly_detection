import os
import time
import math
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from torch import nn, optim
from dataloader import AnomalyDataset, get_loader
from torch.utils.data import Dataset, DataLoader


os.environ["CUDA_VISIBLE_DEVICES"] = str(6)


class LSTM(nn.Module):

    def __init__(self, input_size, hidden_size, output_dim, seq_len, num_layers, bidirectional=False, dropout=0.5):
        super(LSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=num_layers, batch_first=True, dropout=dropout, bidirectional=bidirectional)
        self.num_directions = 2 if bidirectional else 1
        self.fc = nn.Linear(seq_len * hidden_size * self.num_directions, output_dim)
        self.num_layers = num_layers
        self.hidden_size = hidden_size
    
    def init_hidden(self, batch_size):
        # (num_layers * num_directions, batch_size, hidden_size)
        hidden = torch.zeros(self.num_layers * self.num_directions, batch_size, self.hidden_size).cuda()
        cell = torch.zeros(self.num_layers * self.num_directions, batch_size, self.hidden_size).cuda()
        return hidden, cell
        
    def forward(self, x):
        x = x.transpose(0, 1)
        h_0, c_0 = self.init_hidden(x.size(0))
        x, (h_0, c_0) = self.lstm(x, (h_0, c_0))
        # x = x.reshape(x.size(0), -1)
        out = self.fc(x[-1])
        return out


class StackedGRU(nn.Module):
    def __init__(self, input_dim, hidden_dim, n_layers):
        super().__init__()
        self.gru = nn.GRU(
            input_size=input_dim, 
            hidden_size=hidden_dim, 
            num_layers=n_layers, 
            bidirectional=False, 
            dropout=0,
        )
        self.fc = nn.Linear(hidden_dim, input_dim)

    def forward(self, x):
        x = x.transpose(0, 1)
        self.gru.flatten_parameters()
        outs, _ = self.gru(x)
        out = self.fc(outs[-1])
        return x[0] + out


class Encoder(nn.Module):
    def __init__(self, seq_len, n_features, embedding_dim, num_layers):
        super().__init__()
        self.seq_len = seq_len
        self.n_features = n_features
        self.embedding_dim = embedding_dim
        self.hidden_dim = embedding_dim * 2

        self.lstm1 = nn.LSTM(input_size=n_features, hidden_size=self.hidden_dim, num_layers=num_layers, batch_first=True)
        self.lstm2 = nn.LSTM(input_size=self.hidden_dim, hidden_size=embedding_dim, num_layers=num_layers, batch_first=True)

    def forward(self, x):
        batch_size = x.size(0)
        x = x.reshape((batch_size, self.seq_len, self.n_features))
        x, (_, _) = self.lstm1(x)
        x, (h_0, c_0) = self.lstm2(x) # last time step cell

        return h_0.reshape((batch_size, self.embedding_dim)) # (batch_size, 128)


class Decoder(nn.Module):
    def __init__(self, seq_len, embedding_dim, n_features, num_layers):
        super().__init__()
        self.seq_len = seq_len
        self.input_dim = embedding_dim
        self.n_features = n_features
        self.hidden_dim = embedding_dim * 2

        self.lstm1 = nn.LSTM(input_size=embedding_dim, hidden_size=self.hidden_dim, num_layers=num_layers, batch_first=True)
        self.lstm2 = nn.LSTM(input_size=self.hidden_dim, hidden_size=self.hidden_dim, num_layers=num_layers, batch_first=True)
        self.output_layer = nn.Linear(self.hidden_dim, n_features)

    def forward(self, x):
        batch_size = x.size(0)
        x = x.repeat(self.seq_len, self.n_features) # 128 * 140
        x = x.reshape((batch_size, self.seq_len, self.input_dim)) # batch_size * 140 * 128

        x, (h_0, c_0) = self.lstm1(x)
        x, (h_0, c_0) = self.lstm2(x)
        x = x.reshape((batch_size, self.seq_len, self.hidden_dim)) # 140 * 256

        return self.output_layer(x)


class LstmAutoEncoder(nn.Module):
    def __init__(self, seq_len, n_features, embedding_dim, num_layers):
        super().__init__()

        self.encoder = Encoder(seq_len, n_features, embedding_dim, num_layers).cuda()
        self.decoder = Decoder(seq_len, embedding_dim, n_features, num_layers).cuda()

    def forward(self, x):
        x = self.encoder(x) # 1 * 256 -> 256 * 128
        x = self.decoder(x) # 128 * 256 -> 256 * 1

        return x[:, -1, :]


class SAE(nn.Module):
    def __init__(self, input_size):
        # super function is used to use classes of parent class
        super(SAE,self).__init__()
        # by this we can get all the inherited classes of nn.Module
        # first argument is the features, second is the the number of units
        self.fc1 = nn.Linear(input_size, 20)
        self.fc2 = nn.Linear(20, 10) #second layer has 10 neurons
        self.fc3 = nn.Linear(10, 20)
        self.fc4 = nn.Linear(20, input_size)

    def forward(self, x):
        x = torch.sigmoid(self.fc1(x))
        x = torch.tanh(self.fc2(x))
        x = torch.sigmoid(self.fc3(x))
        x = self.fc4(x)
        return x[:, -1, :]


class PositionalEncoding(nn.Module):

    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()       
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        #pe.requires_grad = False
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:x.size(0), :]
          

class TransAm(nn.Module):
    def __init__(self,feature_size,num_layers=1,dropout=0.1):
        super(TransAm, self).__init__()
        self.model_type = 'Transformer'
        
        self.src_mask = None
        self.pos_encoder = PositionalEncoding(feature_size)
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=feature_size, nhead=10, dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)        
        self.decoder = nn.Linear(feature_size, 1)
        self.init_weights()

    def init_weights(self):
        initrange = 0.1    
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, src):
        if self.src_mask is None or self.src_mask.size(0) != len(src):
            device = src.device
            mask = self._generate_square_subsequent_mask(len(src)).to(device)
            self.src_mask = mask

        src = self.pos_encoder(src)
        output = self.transformer_encoder(src, self.src_mask)#, self.src_mask)
        output = self.decoder(output)
        return output[:, -1, :]

    def _generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask


if __name__ == "__main__":
    device = "cuda:6"
    # model = SAE(1)
    # model = LstmAutoEncoder(1, 1, 32, 1)
    # model = StackedGRU(1, 128, 3)
    # model = LSTM(1, 2, 1, 1, 1, bidirectional=True)
    model.cuda()
    print(model)

    tr_path = "/daintlab/data/anomaly_detection/NAB-master/data/realAWSCloudwatch/rds_cpu_utilization_e47b3b.csv"
    tst_path = "/daintlab/data/anomaly_detection/NAB-master/data/realAWSCloudwatch/rds_cpu_utilization_cc0c53.csv"
    tr_filename = 'realAWSCloudwatch/rds_cpu_utilization_e47b3b.csv'

    tr_ds = AnomalyDataset(tr_path, tr_path, tr_filename, tr_filename)
    tr_dl = DataLoader(tr_ds, shuffle=False, batch_size=128) 
    x = next(iter(tr_dl))
    output = model(x.cuda())
    # print(x.size())
    print(output.size())
