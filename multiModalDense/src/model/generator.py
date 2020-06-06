import torch.nn as nn
import torch.nn.functional as F
import torch

class Generator(nn.Module):
    """
        Generator has 2 linear layers to concatinate all modality outputs and generate softmax output. This is known as late fusion.
        Using generator to Concatenate outputs from subtitle, audio and video decoders. This is output module in transformer architecture.
    """
    def __init__(self, subtitle_modal_dimension, audio_modal_dimension, video_modal_dimension, caption_vocab_size, dropout_percentage):
        """
            Creates a linear model of dimension (sum_of_all_modality_outputs, caption_vocab_size) and adds dropout between final linear layer of dimension
            (caption_vocab_size, caption_vocab_size).
            Args:
                subtitle_modal_dimension: output dimension for subtitles model
                audio_modal_dimension: output dimension for audio model
                video_modal_dimension: output dimension for video model
                caption_vocab_size: total vocabulary size of all training captions
                dropout_percentage: dropout percentage between 2 linear layers
        """
        super(Generator, self).__init__()
        self.linear = nn.Linear(subtitle_modal_dimension + audio_modal_dimension + video_modal_dimension, caption_vocab_size)
        self.dropout = nn.Dropout(dropout_percentage)
        self.linear2 = nn.Linear(caption_vocab_size, caption_vocab_size)
        
    def forward(self, subtitle_model_output, audio_model_output, video_model_output):
        """
            Following steps are performed,
            1. Concatenate outputs from subtitle, audio and video encoder-decoder network on last dimension. Assuming other dimesions are same.
            2. Apply linear layer to the output of above step
            3. Apply relu activate, dropout and another linear layer to output of above step
            4. Finally apply softmax function to last dimension to generate probabilites for all vocabulary of captions
            Args:
                subtitle_model_output: Output from subtitles model of shape (batch_size, sequence_length, model_dimension)
                audio_model_output: Output from audio model of shape (batch_size, sequence_length, model_dimension)
                video_model_output: Output from video model of shape (batch_size, sequence_length, model_dimension)
            Returns:
                Softmax applied output of shape (batch_size, sequence_length, caption_vocab_size)
        """
        x = torch.cat([subtitle_model_output, audio_model_output, video_model_output], dim=-1)
        x = self.linear(x)
        x = self.linear2(self.dropout(F.relu(x)))
        
        return F.log_softmax(x, dim=-1)
