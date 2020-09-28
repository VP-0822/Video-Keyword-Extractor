import torch.nn as nn
import torch.nn.functional as F
import torch

class CommonGenerator(nn.Module):
    """
        Generator has 2 linear layers to concatinate all modality outputs and generate softmax output. This is known as late fusion.
        Using generator to Concatenate outputs from subtitle, audio and video decoders. This is output module in transformer architecture.
    """
    def __init__(self, video_modal_dimension, caption_vocab_size, dropout_percentage):
        """
            Creates a linear model of dimension (sum_of_all_modality_outputs, caption_vocab_size) and adds dropout between final linear layer of dimension
            (caption_vocab_size, caption_vocab_size).
            Args:
                video_modal_dimension: output dimension for video model
                caption_vocab_size: total vocabulary size of all training captions
                dropout_percentage: dropout percentage between 2 linear layers
        """
        super(CommonGenerator, self).__init__()
        self.linear = nn.Linear(video_modal_dimension, caption_vocab_size)
        self.dropout = nn.Dropout(dropout_percentage)
        self.linear2 = nn.Linear(caption_vocab_size, caption_vocab_size)
        
    def forward(self, video_model_output):
        """
            Following steps are performed,
            1. Concatenate outputs from subtitle, audio and video encoder-decoder network on last dimension. Assuming other dimesions are same.
            2. Apply linear layer to the output of above step
            3. Apply relu activate, dropout and another linear layer to output of above step
            4. Finally apply softmax function to last dimension to generate probabilites for all vocabulary of captions
            Args:
                video_model_output: Output from video model of shape (batch_size, sequence_length, model_dimension)
            Returns:
                Softmax applied output of shape (batch_size, sequence_length, caption_vocab_size)
        """
        x = video_model_output
        x = self.linear(x)
        x = self.linear2(self.dropout(F.relu(x)))
        
        return F.log_softmax(x, dim=-1)
