import torch.nn as nn
from model.modelUtil import clone
from model.multiheadedAttention import MultiheadedAttention
from model.residualConnection import ResidualConnection
from model.feedforward import PositionwiseFeedForward

class EncoderLayer(nn.Module):
    """
        Encoder Layer of the transformer model. Encoder Layer is used to encode Video, Audio and Subtitles inputs individually. 
    """
    def __init__(self, model_dimension, subtitle_relu_model_dimension, audio_relu_model_dimension, dropout_percentage, number_of_heads, feedforward_dimension):
        """
            Creates 2 copies of ResidualConnection, multiheaded attention with number_of_heads heads and fully-connected layer of shape (model_dimension, feedforward_dimension)
            Args:
                model_dimension: model dimension generally same as embedding_dimension of VocabEmbedding or FeatureEmbedding
                dropout_percentage: droupout percentage for residual connection
                number_of_heads: number of heads for multiheaded attention
                feedforward_dimension: units of feedforward layer. Generally 2048 units
        """
        super(EncoderLayer, self).__init__()
        self.res_layers = clone(ResidualConnection(model_dimension, dropout_percentage), 4)
        self.self_att = MultiheadedAttention(model_dimension, number_of_heads)
        print('model dimension: ' + str(model_dimension))
        print('subtitle model dimension: ' + str(subtitle_relu_model_dimension))
        print('audio model dimension: ' + str(audio_relu_model_dimension))
        print('===============')
        self.fc1 = nn.Linear(subtitle_relu_model_dimension, model_dimension)
        self.fc2 = nn.Linear(audio_relu_model_dimension, model_dimension)
        self.contextual_attention = MultiheadedAttention(model_dimension, number_of_heads)
        self.feed_forward = PositionwiseFeedForward(model_dimension, feedforward_dimension)
        
    def forward(self, x, subtitle_modality_relu, audio_modality_relu, source_mask, subtitle_source_mask): # x - (B, seq_len, d_model) src_mask (B, 1, S)
        """
            Following steps are performed to encode input sequence
            1. Lambda function for performing self-attention is created
            2. self-attention and inputs are applied residual connection to obtained intermediate output
            3. Output obtained from above step is used to pass to residual connection with sublayer as position-wise fully connected network

            Args:
                x: Input with shape (batch_size, sequence_length, model_dimension)
                source_mask: Source mask of dimension (batch_size, 1, sequence_length)
            Returns:
                encoder output of shape (batch_size, sequence_length, model_dimension)
        """
        # sublayer should be a function which inputs x and outputs transformation
        # thus, lambda is used instead of just `self.self_att(x, x, x)` which outputs 
        # the output of the self attention refer http://nlp.seas.harvard.edu/2018/04/03/attention.html
        sublayer0 = lambda x: self.self_att(x, x, x, source_mask) # Query, Key and Value are same for self-attention
        subtitle_encoder_output = self.fc1(subtitle_modality_relu)
        sublayer1 = lambda x: self.contextual_attention(x, subtitle_encoder_output, subtitle_encoder_output, subtitle_source_mask)
        audio_encoder_output = self.fc2(audio_modality_relu)
        sublayer2 = lambda x: self.contextual_attention(x, audio_encoder_output, audio_encoder_output, source_mask)
        sublayer3 = self.feed_forward
        x = self.res_layers[0](x, sublayer0)
        x = self.res_layers[1](x, sublayer1)
        x = self.res_layers[2](x, sublayer2)
        x = self.res_layers[3](x, sublayer3)
        
        return x
    
class Encoder(nn.Module):
    """
        Layered Encoders are used to apply deep learning. In which the first layer produces internal represented output which is used as input by subsequent encoder layer
        and produces new output. Number of layers are derived from number_of_layers parameter. 
    """
    
    def __init__(self, model_dimension, subtitle_model_dimension, audio_model_dimension, dropout_percentage, number_of_heads, feedforward_dimension, number_of_layers):
        """
            Create EncoderLayer copy number_of_layers times.
            Args:
                model_dimension: model dimension generally same as embedding_dimension of VocabEmbedding or FeatureEmbedding
                dropout_percentage: droupout percentage for residual connection
                number_of_heads: number of heads for multiheaded attention
                feedforward_dimension: units of feedforward layer. Generally 2048 units
                number_of_layers: Number of encoder layers
        """
        super(Encoder, self).__init__()
        self.enc_layers = clone(EncoderLayer(model_dimension, subtitle_model_dimension, audio_model_dimension, dropout_percentage, number_of_heads, feedforward_dimension), number_of_layers)
        
    def forward(self, x, sub_modality_relu, audio_modality_relu, source_mask, subtitle_source_mask):
        """
            Input is passed through first_layer and output of each layer is passed as input to next layer. This is done to apply deep learning by creating dense network.
            Args:
                x: Input with shape (batch_size, sequence_length, model_dimension)
                sub_modality_relu: Input for subtitle with shape (batch_size, sequence_length, model_dimension)
                audio_modality_relu: Input for audio with shape (batch_size, sequence_length, model_dimension)
                source_mask: Source mask of dimension (batch_size, 1, sequence_length)
            Returns:
                encoder output of shape (batch_size, sequence_length, model_dimension)
        """
        for layer in self.enc_layers:
            x = layer(x, sub_modality_relu, audio_modality_relu, source_mask, subtitle_source_mask)
        
        return x