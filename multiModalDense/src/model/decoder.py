import torch.nn as nn
from model.modelUtil import clone
from model.multiheadedAttention import MultiheadedAttention
from model.residualConnection import ResidualConnection
from model.feedforward import PositionwiseFeedForward

class DecoderLayer(nn.Module):
    """
        Decoder Layer of the transformer model. Decoder Layer is used to decode encoded Video, Audio and Subtitles inputs individually into output tensors for generator module. 
    """
    def __init__(self, model_dimension, dropout_percentage, number_of_heads, feedforward_dimension):
        """
            1. Creates 3 copies of ResidualConnection
            2. multiheaded attention with number_of_heads heads for self attention and encoder-decoder attention
            3. fully-connected layer of shape (model_dimension, feedforward_dimension)
            Args:
                model_dimension: model dimension generally same as embedding_dimension of VocabEmbedding or FeatureEmbedding
                dropout_percentage: droupout percentage for residual connection
                number_of_heads: number of heads for multiheaded attention
                feedforward_dimension: units of feedforward layer. Generally 2048 units
        """

        super(DecoderLayer, self).__init__()
        self.res_layers = clone(ResidualConnection(model_dimension, dropout_percentage), 3)
        self.self_att = MultiheadedAttention(model_dimension, number_of_heads)
        self.enc_att = MultiheadedAttention(model_dimension, number_of_heads)
        self.feed_forward = PositionwiseFeedForward(model_dimension, feedforward_dimension)
        
    def forward(self, x, encoder_memory, source_mask, target_mask): # x, memory - (B, seq_len, d_model) src_mask (B, 1, S) trg_mask (B, S, S)
        """
            Following steps are performed to decode encoded data
            1. Lambda function for performing self-attention is created for decoder inputs
            2. Lambda function for performin encoder-decoder is created for encoder-memory and decoder intermediate state
            2. self-attention and inputs are applied residual connection to obtained intermediate output
            3. Output obtained from above step is used to pass to residual connection with sublayer as position-wise fully connected network

            Args:
                x: Input with shape (batch_size, sequence_length, model_dimension)
                encoder_memory: encoder output as memory of shape (batch_size, sequence_length, model_dimension)
                source_mask: Source mask of dimension (batch_size, 1, sequence_length)
                target_mask: Target mask of dimension (batch_size, sequence_length, sequence_length)
            Returns:
                decoder output of shape (batch_size, sequence_length, model_dimension)
        """
        sublayer0 = lambda x: self.self_att(x, x, x, target_mask) # Query, Key and Value are same for self-attention
        sublayer1 = lambda x: self.enc_att(x, encoder_memory, encoder_memory, source_mask) # Query is decoder intermediate output, Key and Value are encoder-output for encoder-decoder attention
        sublayer2 = self.feed_forward
        
        x = self.res_layers[0](x, sublayer0)
        x = self.res_layers[1](x, sublayer1)
        x, _ = self.res_layers[2](x, sublayer2, True)
        
        return x
    
class Decoder(nn.Module):
    """
        Layered Decoders are used to apply deep learning. In which the first layer produces internal represented output which is used as input by subsequent decoder layer
        and produces new output. Number of layers are derived from number_of_layers parameter. 
    """
    
    def __init__(self, model_dimension, dropout_percentage, number_of_heads, feedforward_dimension, number_of_layers):
        """
            Create DecoderLayer copy number_of_layers times.
            Args:
                model_dimension: model dimension generally same as embedding_dimension of VocabEmbedding or FeatureEmbedding
                dropout_percentage: droupout percentage for residual connection
                number_of_heads: number of heads for multiheaded attention
                feedforward_dimension: units of feedforward layer. Generally 2048 units
                number_of_layers: Number of encoder layers
        """
        super(Decoder, self).__init__()
        self.dec_layers = clone(DecoderLayer(model_dimension, dropout_percentage, number_of_heads, feedforward_dimension), number_of_layers)
        
    def forward(self, x, encoder_memory, source_mask, target_mask):
        """
            Input is passed through first_layer and output of each layer is passed as input to next layer. This is done to apply deep learning by creating dense network.

            Args:
                x: Input with shape (batch_size, sequence_length, model_dimension)
                encoder_memory: encoder output as memory of shape (batch_size, sequence_length, model_dimension)
                source_mask: Source mask of dimension (batch_size, 1, sequence_length)
                target_mask: Target mask of dimension (batch_size, sequence_length, sequence_length)
            Returns:
                decoder output of shape (batch_size, sequence_length, model_dimension)
        """
        for layer in self.dec_layers:
            x = layer(x, encoder_memory, source_mask, target_mask)
        return x