import numpy as np
import torch.nn as nn
import torch

class PositionalEncoder(nn.Module):
    """
        This class is to apply positional encoding to input audio, video and text inputs. Since in this implementation we are not using
        recurrent networks hence real sense of order in the input sequence is missing, which is taken care by Positional Encoding Layer
        by applying alternately sine and cosine funtion and output of this model is applied to embedded inputs.
        Referred from "Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez, Ł ukasz Kaiser, and Illia Polosukhin. 
        Attention is all you need. In NeurIPS, 2017." https://arxiv.org/pdf/1706.03762.pdf
    """
    
    def __init__(self, embedding_dimension, dropout_percentage, sequence_length=3660):
        """
            sin function is applied on every odd position in input sequence and cosine function is applied on every even position in input sequence
            Args:
                embedding_dimension: feature_dimension of inputs after embedding
                dropout_percentage: dropout percentage from normalised between 0 to 1. To be used in nn.Dropout layer. Refer http://jmlr.org/papers/v15/srivastava14a.html
                sequence_length: Sequence length of the various inputs (Depends where this layer is applied). default longest sequence length is 3660.
        """
        super(PositionalEncoder, self).__init__()
        self.embedding_dimension = embedding_dimension
        self.dropout = nn.Dropout(dropout_percentage)
        
        # position encoding matrix
        pos_enc_mat = np.zeros((sequence_length, embedding_dimension))
        odds = np.arange(0, embedding_dimension, 2) # create vector of half of size of embedding_dimension
        evens = np.arange(1, embedding_dimension, 2) # create vector of half of size of embedding_dimension

        for pos in range(sequence_length):
            pos_enc_mat[pos, odds] = np.sin(pos / (10000 ** (odds / embedding_dimension)))
            pos_enc_mat[pos, evens] = np.cos(pos / (10000 ** (evens / embedding_dimension)))
        
        # create 3D position matrix of shape (1, sequence_length, embedding_dimension)
        self.pos_enc_mat = torch.from_numpy(pos_enc_mat).unsqueeze(0)
        
    def forward(self, x):
        """
            Element wise summation is performed with position encoded 3D model with input 3D model
            Dimension of position wise 3D matrix is (1, sequence_length, embedding_dimension).
            After applying positional encoding, dropout is applied to avoid overfitting.
            Args:
                x: Dimension of input after embedding layer is (batch_size, sequence_length, embedding_dimension)
            Returns:
                position encoded input of size (batch_size, sequence_length, embedding_dimension)
        """
        batch_size, sequence_length, embedding_dimension = x.shape
        
        # Only consider positional elements until sequence_length positions
        x = x + self.pos_enc_mat[:, :sequence_length, :].type_as(x)

        # apply dropout to position encoded inputs
        x = self.dropout(x)
        
        return x # same as input