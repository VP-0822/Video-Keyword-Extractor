import torch.nn as nn
from modelUtil import clone, attention

class MultiheadedAttention(nn.Module):
    """
        Multiheaded Attention class. Read more in paper https://arxiv.org/pdf/1706.03762.pdf. 
        And specific to video captioning task from paper 
        @InProceedings{MDVC_Iashin_2020,
            author = {Iashin, Vladimir and Rahtu, Esa},
            title = {Multi-modal Dense Video Captioning},
            booktitle = {Workshop on Multimodal Learning (CVPR Workshop)},
            year = {2020}
            }
    """
    
    def __init__(self, model_dimension, number_of_heads):
        """
            Creates 4 copies of linear layer for Query, Key, Value and attention connections.
            It will help attention to have multiple “representation subspaces” to focus on. This number is equivalent to number_of_heads.
            4 Linear layers are used as weights for Query, Key, Value and multihead concatinated attention.
            For reducing computational cost, weights (from linear layers) are shared between all the heads. 
            Args:
                model_dimension: model dimension
                number_of_heads: number of heads in mutliheaded attention
        """
        super(MultiheadedAttention, self).__init__()
        assert model_dimension % number_of_heads == 0
        self.model_dimension = model_dimension
        self.number_of_heads = number_of_heads
        self.d_k = model_dimension // number_of_heads
        self.linears = clone(nn.Linear(model_dimension, model_dimension), 4) # bias True??
        
    def forward(self, Q, K, V, mask): # Q, K, V are of size (B, seq_len, d_model)
        """
            Following steps are performed.
            1. Apply linear layer to input Q, K and V. To have trainable weights.
            2. Reshape Q, K and V to appropriate shape of (batch_size, runtime_sequence_length, number_of_heads, d_k) After applying transpose (batch_size, number_of_heads, runtime_sequence_length, d_k)
            3. Apply attention using Q, K, V and mask matrix
            4. Reshape attention matrix obtained from above step to appropriate shape (batch_size, sequence_length, model_dimension)
            5. Apply final linear layer to obtained final attention matrix. 

            For self attention Q, K, and V have same inputs.
            Args:
                Q: Query matrix of shape (batch_size, sequence_length, model_dimension)
                K: Key matrix of shape (batch_size, sequence_length, model_dimension)
                V: Value matrix of shape (batch_size, sequence_length, model_dimension)
                mask: Mask matrix of shape (batch_size, 1, sequence_length)
            Returns:
                Multiheaded attention tensor of shape (batch_size, sequence_length, model_dimension)
        """
        B, seq_len, d_model = Q.shape
        
        Q = self.linears[0](Q) # (batch_size, sequence_length, model_dimension) -> (batch_size, sequence_length, model_dimension) => input_linear_model_dim = output_linear_model_dim
        K = self.linears[1](K) # (batch_size, sequence_length, model_dimension) -> (batch_size, sequence_length, model_dimension) => input_linear_model_dim = output_linear_model_dim
        V = self.linears[2](V) # (batch_size, sequence_length, model_dimension) -> (batch_size, sequence_length, model_dimension) => input_linear_model_dim = output_linear_model_dim
        
        Q = Q.view(B, -1, self.H, self.d_k).transpose(-3, -2) # reshaped to (batch_size, number_of_heads, runtime_sequence_length, d_k)
        K = K.view(B, -1, self.H, self.d_k).transpose(-3, -2) # reshaped to (batch_size, number_of_heads, runtime_sequence_length, d_k)
        V = V.view(B, -1, self.H, self.d_k).transpose(-3, -2) # reshaped to (batch_size, number_of_heads, runtime_sequence_length, d_k)
        
        if mask is not None:
            # the same mask for all heads
            mask = mask.unsqueeze(1) # (batch_size, 1, 1, sequence_length)
        
        att = attention(Q, K, V, mask) # returns (batch_size, number_of_heads, runtime_sequence_length, d_k)
        att = att.transpose(-3, -2).contiguous().view(B, seq_len, d_model) # reshaped to (batch_size, sequence_length, model_dimension)
        att = self.linears[3](att) # (batch_size, sequence_length, model_dimension)
        
        return att # (batch_size, sequence_length, model_dimension)