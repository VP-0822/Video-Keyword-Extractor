import torch.nn as nn
from model.modelUtil import clone
from model.multiheadedAttention import MultiheadedAttention
from model.residualConnection import ResidualConnection
from model.feedforward import PositionwiseFeedForward, PositionwiseFeedForwardDecoder
import torch

class CommonDecoderLayer(nn.Module):
    """
        Decoder Layer of the transformer model. Decoder Layer is used to decode encoded Video, Audio and Subtitles inputs individually into output tensors for generator module. 
    """
    def __init__(self, model_dimension, dropout_percentage, number_of_heads, feedforward_dimension, audio_modal_dimension, subtitile_modal_dimension):
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

        super(CommonDecoderLayer, self).__init__()
        # self.self_att = MultiheadedAttention(model_dimension + audio_modal_dimension + subtitile_modal_dimension, number_of_heads)
        # self.self_att_res_layer = ResidualConnection(model_dimension + audio_modal_dimension + subtitile_modal_dimension, dropout_percentage)

        # self.enc_att = MultiheadedAttention(model_dimension + audio_modal_dimension + subtitile_modal_dimension, number_of_heads)
        # self.enc_att_res_layer = ResidualConnection(model_dimension + audio_modal_dimension + subtitile_modal_dimension, dropout_percentage)

        # self.feed_forward = PositionwiseFeedForward(model_dimension + audio_modal_dimension + subtitile_modal_dimension, feedforward_dimension)
        # self.feed_forward_res_layer = ResidualConnection(model_dimension + audio_modal_dimension + subtitile_modal_dimension, dropout_percentage)
        self.res_layers = clone(ResidualConnection(model_dimension, dropout_percentage), 5)
        self.self_att = MultiheadedAttention(model_dimension, number_of_heads)
        self.video_enc_att = MultiheadedAttention(model_dimension, number_of_heads)
        self.audio_enc_att = MultiheadedAttention(model_dimension, number_of_heads)
        self.subtitle_enc_att = MultiheadedAttention(model_dimension, number_of_heads)
        self.feed_forward = PositionwiseFeedForwardDecoder(model_dimension, feedforward_dimension, model_dimension + audio_modal_dimension + subtitile_modal_dimension)
        self.audio_linear = nn.Linear(audio_modal_dimension, model_dimension)
        self.subtitle_linear = nn.Linear(subtitile_modal_dimension, model_dimension)
        
    def forward(self, x, video_encoder_memory, audio_encoder_memory, subtitle_encoder_memory, source_mask, target_mask, subtitle_mask): # x, memory - (B, seq_len, d_model) src_mask (B, 1, S) trg_mask (B, S, S)
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
        # print(video_encoder_memory.size())
        # print(audio_encoder_memory.size())
        # print(subtitle_encoder_memory.size())
        # combined_memory = torch.cat([video_encoder_memory, audio_encoder_memory, subtitle_encoder_memory], dim=-1)
        # sublayer0 = lambda x: self.self_att(x, x, x, target_mask)
        # sublayer1 = lambda x: self.self_att(x, combined_memory, combined_memory, source_mask)
        # sublayer2 = self.feed_forward
        
        sublayer0 = lambda x: self.self_att(x, x, x, target_mask) # Query, Key and Value are same for self-attention
        subtitle_modified_memory = self.subtitle_linear(subtitle_encoder_memory)
        sublayer1 = lambda x: self.subtitle_enc_att(x, subtitle_modified_memory, subtitle_modified_memory, subtitle_mask) # Query is decoder intermediate output, Key and Value are encoder-output for encoder-decoder attention
        audio_modified_memory = self.audio_linear(audio_encoder_memory)
        sublayer2 = lambda x: self.audio_enc_att(x, audio_modified_memory, audio_modified_memory, source_mask) # Query is decoder intermediate output, Key and Value are encoder-output for encoder-decoder attention
        sublayer3 = lambda x: self.video_enc_att(x, video_encoder_memory, video_encoder_memory, source_mask) # Query is decoder intermediate output, Key and Value are encoder-output for encoder-decoder attention
        sublayer4 = self.feed_forward
        
        x = self.res_layers[0](x, sublayer0)
        x = self.res_layers[1](x, sublayer1)
        x = self.res_layers[2](x, sublayer2)
        x = self.res_layers[3](x, sublayer3)
        x = self.res_layers[4](x, sublayer4)

        # x = self.self_att_res_layer(x, sublayer0)
        # x = self.enc_att_res_layer(x, sublayer1)
        # x = self.feed_forward_res_layer(x, sublayer2)
        return x
    
class CommonDecoder(nn.Module):
    """
        Layered Decoders are used to apply deep learning. In which the first layer produces internal represented output which is used as input by subsequent decoder layer
        and produces new output. Number of layers are derived from number_of_layers parameter. 
    """
    
    def __init__(self, model_dimension, dropout_percentage, number_of_heads, feedforward_dimension, number_of_layers, audio_modal_dimension, subtitile_modal_dimension):
        """
            Create DecoderLayer copy number_of_layers times.
            Args:
                model_dimension: model dimension generally same as embedding_dimension of VocabEmbedding or FeatureEmbedding
                dropout_percentage: droupout percentage for residual connection
                number_of_heads: number of heads for multiheaded attention
                feedforward_dimension: units of feedforward layer. Generally 2048 units
                number_of_layers: Number of encoder layers
        """
        super(CommonDecoder, self).__init__()
        self.dec_layers = clone(CommonDecoderLayer(model_dimension, dropout_percentage, number_of_heads, feedforward_dimension, audio_modal_dimension, subtitile_modal_dimension), number_of_layers)
        
    def forward(self, x, video_encoder_memory, audio_encoder_memory, subtitle_encoder_memory, source_mask, target_mask, subtitle_mask):
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
            x = layer(x, video_encoder_memory, audio_encoder_memory, subtitle_encoder_memory, source_mask, target_mask, subtitle_mask)
        return x