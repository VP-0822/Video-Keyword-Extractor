import torch.nn as nn
import torch.nn.functional as F

class PositionwiseFeedForward(nn.Module):
    """
        Position-wise fully connected feed forward module class. Generally used at the end of encoder and decoder each.
    """
    
    def __init__(self, model_dimension, feedforward_dimension):
        """
            Creates 2 Linear Layers which are treated as fully connected layers of following shape.
            FC1 = shape (model_dimension, feedforward_dimension)
            FC2 = shape (feedforward_dimension, model_dimension)
            Args:
                model_dimension: model dimension generally same as embedding_dimension of VocabEmbedding or FeatureEmbedding
                feedforward_dimension: units of feedforward layer. Generally 2048 units
        """
        super(PositionwiseFeedForward, self).__init__()
        self.model_dimension = model_dimension
        self.feedforward_dimension = feedforward_dimension
        self.fc1 = nn.Linear(model_dimension, feedforward_dimension)
        self.fc2 = nn.Linear(feedforward_dimension, model_dimension)
        
    def forward(self, x): # x - (B, seq_len, d_model)
        """
            Creates 2 layer FC layer with relu activation function.
            Args:
                x: input with shape (batch_size, sequence_length, model_dimension)
            Returns:
                output with same dimension as input (batch_size, sequence_length, model_dimension)
        """
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        
        return x