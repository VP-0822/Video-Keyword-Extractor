import torch.nn as nn

class ResidualConnection(nn.Module):
    """
        Class for residual connection to skip attention or position-wise fully connected network. Also known as skip connections
    """

    def __init__(self, input_shape_without_batchsize, dropout_percentage):
        """
            Creates Layer normalization and dropout layer.
            Args:
                input_shape_without_batchsize: input shape without batch_size to create layer normalization. shape (sequence_length, model_dimension)
                dropout_percentage: Dropout percentage to avoid overfitting
        """
        super(ResidualConnection, self).__init__()
        self.norm = nn.LayerNorm(input_shape_without_batchsize)
        self.dropout = nn.Dropout(dropout_percentage)
        
    def forward(self, x, sublayer, multireturn=False):
        """
            Apply residual connection around passed sublayer. This involves operation of applying sublayer after layerNorm and at the end element-wise summation of
            residual connection (Actual input) and output of sublayer.
            Args:
                x: input of shape (batch_size, sequence_length, model_dimension)
                sublayer: nn.Module instance of either Attention module or Position-wise feedforward module.
            Returns:
                output of shape (batch_size, sequence_length, model_dimension)
        """
        res = self.norm(x)
        if multireturn is False:
            res = sublayer(res)
            res = self.dropout(res)
            return res

        res, sublayer_res = sublayer(res)
        res = self.dropout(res)
        
        return x + res, sublayer_res