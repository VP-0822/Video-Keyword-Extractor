import numpy as np
import torch.nn as nn

class VocabularyEmbedder(nn.Module):
    """
        This class is used for caption and subtitle embedding purpose
    """

    def __init__(self, total_vocab_size, embedding_dimension):
        """
            Embedding layer has shape (total_vocab_size, embedding_dimension)
            Args:
                voc_size: total vocabulary size
                embedding_dimension: Embedding dimension, usually it should be model input dimension
        """
        super(VocabularyEmbedder, self).__init__()
        self.embedding_dimension = embedding_dimension
        self.embedder = nn.Embedding(total_vocab_size, embedding_dimension)
    
    def forward(self, x):
        """
            Multiplying output from Torch Embedding layer with square root of embedding dimension. This technique is
            referenced from "Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez, Ł ukasz Kaiser, and Illia Polosukhin. 
            Attention is all you need. In NeurIPS, 2017." https://arxiv.org/pdf/1706.03762.pdf
            Args:
                x: Vocabulary tokens of size (batch_size, sequence_length)
            Returns: 
                shape (batch_size, sequence_length, embedding_dimension)
        """
        x = self.embedder(x)
        x = x * np.sqrt(self.embedding_dimension)
        
        return x
    
class FeatureEmbedder(nn.Module):
    """
        This embedder class is used to apply dense embedding to non-text featues such as video and audio.
        This technique is referred from https://arxiv.org/pdf/1706.03762.pdf known as linear feature embedding
    """
    
    def __init__(self, feature_dimension, embedding_dimension):
        """
            Linear nueral network model is used for embedding purpose of dimension (feature_dimension, embedding_dimension)
            Args:
                feature_dimension: Dimension of feature vector. For example video feature dimension is 1024 for video frames
                embedding_dimension: Dimension of embedding vector for non-text features A.K.A model_dimension
        """
        super(FeatureEmbedder, self).__init__()
        self.embedding_dimension = embedding_dimension
        self.embedder = nn.Linear(feature_dimension, embedding_dimension)
        
    def forward(self, x):
        """
            Multiplying output from Torch Linear layer (used as embedding layer) with square root of embedding dimension. This technique is
            referenced from "Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez, Ł ukasz Kaiser, and Illia Polosukhin. 
            Attention is all you need. In NeurIPS, 2017." https://arxiv.org/pdf/1706.03762.pdf
            Args:
                x: Feature tensor of shape (batch_size, sequence_length, feature_dimension)
            Returns: 
                shape (batch_size, sequence_length, embedding_dimension)
        """
        x = self.embedder(x)
        x = x * np.sqrt(self.d_model)
        
        return x