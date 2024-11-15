import torch
import torch.nn as nn
import torch.nn.functional as F

class TrajectoryNetwork(nn.Module):
    def __init__(self, LSTM_config):
        super(TrajectoryNetwork, self).__init__()

        self.LSTM_config = LSTM_config

        self.enable_cuda = LSTM_config['enable_cuda']

        self.encode_size = LSTM_config['encode_size']

        self.input_embedding_size = LSTM_config['input_embedding_size']

        self.decode_size = LSTM_config['decode_size']

        self.output_size = LSTM_config['predict_step']

        self.input_embedding = nn.Linear(7, self.input_embedding_size)      # 'global time', 'Local_X', 'Local_Y', 'v_Vel', 'v_Acc', 'Theta', 'Movement'

        self.ego_history_lstm = nn.LSTM(self.input_embedding_size, self.encode_size, 1)

        self.neighbor_history_lstm = nn.LSTM(self.input_embedding_size, self.encode_size, 1)

        self.decode_lstm = nn.LSTM(self.encode_size, self.decode_size)

        self.output = torch.nn.Linear(self.decode_size, 2)

        self.pre4att = nn.Sequential(
            nn.Linear(self.encode_size, 1),
        )

        self.tanh = nn.Tanh()

        self.leaky_relu = nn.LeakyReLU(0.1)

    def attention(self, lstm_out_weight, lstm_out):
       
        alpha = F.softmax(lstm_out_weight, 1) 

        lstm_out = lstm_out.permute(0, 2, 1)
         
        new_hidden_state = torch.bmm(lstm_out, alpha).squeeze(2) 
        new_hidden_state = F.relu(new_hidden_state)


        return new_hidden_state, alpha

    
    def forward(self, history_states, neighbors_his_states, mask, enable_neighbor=True):
        embedding_input = self.leaky_relu(self.input_embedding(history_states))

        history_lstm_out, (history_hidden, _) = self.ego_history_lstm(embedding_input)

        # history_lstm_out = history_lstm_out.permute(1, 0, 2) 
        history_weight = self.pre4att(self.tanh(history_lstm_out))

        new_history_hidden, soft_attn_weights = self.attention(history_weight, history_lstm_out) 
        new_history_hidden = new_history_hidden.unsqueeze(2) 

        if enable_neighbor:
            embedding_neighbors = self.leaky_relu(self.input_embedding(neighbors_his_states))
            neighbor_lstm_out, (neighbor_hidden, _) = self.neighbor_history_lstm(embedding_neighbors)
            # neighbor_lstm_out = neighbor_lstm_out.permute(1, 0, 2)
            neighbors_weight = self.pre4att(self.tanh(neighbor_lstm_out))
            new_neighbors_hidden, soft_neighbor_attn_weights = self.attention(neighbors_weight, neighbor_lstm_out) 

            neighbor_pos_encode = torch.zeros_like(mask).float()

            mask = mask.bool()
            neighbor_pos_encode = neighbor_pos_encode.masked_scatter_(mask, new_neighbors_hidden)

            neighbor_pos_encode = neighbor_pos_encode.permute(0,3,2,1)

            neighbor_pos_encode = neighbor_pos_encode.contiguous().view(neighbor_pos_encode.shape[0], neighbor_pos_encode.shape[1], -1)

            combine_encode = torch.cat((new_history_hidden, neighbor_pos_encode), 2)
        else:
            combine_encode = new_history_hidden

        combine_encode = combine_encode.permute(0, 2, 1)

        weight = self.pre4att(self.tanh(combine_encode))

        all_encode, _ = self.attention(weight, combine_encode)
        # print("all_encode", all_encode.shape)

        predict = self.decode(all_encode)

        return predict
    
    def decode(self, encoded_pred):
        encoded_pred = encoded_pred.repeat(self.output_size, 1, 1)

        h_dec, (_, _) = self.decode_lstm(encoded_pred) 
        h_dec = h_dec.permute(1, 0, 2) 
        fut_pred = self.output(h_dec) 
        # fut_pred = fut_pred.permute(2, 1, 0) 
        fut_pred = self.outputActivation(fut_pred)
        return fut_pred
    
    def outputActivation(self, x):
        muX = x[:,:,0:1]
        muY = x[:,:,1:2]

        out = torch.cat([muX, muY],dim=2)
        return out
        