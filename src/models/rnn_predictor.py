import torch
import torch.nn as nn
import numpy as np

class IOPredictor(nn.Module):
    """LSTM-based I/O pattern predictor optimized for Intel CPU"""
    
    def __init__(self, input_size=128, hidden_size=256, num_layers=2):
        super(IOPredictor, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # LSTM for sequence learning
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.2 if num_layers > 1 else 0
        )
        
        # Output layer
        self.fc = nn.Linear(hidden_size, input_size)
        
        # Intel CPU optimization: use float32 for better performance
        self.dtype = torch.float32
        
    def forward(self, x, hidden=None):
        # Ensure input is float32 for Intel CPU optimization
        x = x.to(self.dtype)
        
        # LSTM forward pass
        lstm_out, hidden = self.lstm(x, hidden)
        
        # Get last output
        last_output = lstm_out[:, -1, :]
        
        # Predict next blocks
        prediction = self.fc(last_output)
        
        return prediction, hidden
    
    def predict_next_blocks(self, io_history, top_k=10):
        """Predict top-k blocks to be accessed next"""
        self.eval()
        with torch.no_grad():
            prediction, _ = self.forward(io_history)
            probabilities = torch.softmax(prediction, dim=-1)
            top_values, top_indices = torch.topk(probabilities, k=min(top_k, prediction.size(-1)))
            return top_indices.numpy(), top_values.numpy()
