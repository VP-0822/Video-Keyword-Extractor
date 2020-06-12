import torch
import numpy as np
from copy import deepcopy
import torch.nn.functional as F
import torch.nn as nn

"""
    Provides utility functions needed for transformer model
"""

def subsequent_mask(sequence_length):
    """
        Apply subsequent masking so that we can train variable length inputs.
        Subsequent masking is used for target as input in decoder, to prevent attention for captions looking into future. 
        This is done by applying masking on upper triangle of input matrix.
        Args:
            sequence_length: sequence length of the input
        Returns:
            mask matrix of shape (1, sequence_length, sequence_length)
    """
    mask = torch.ones(1, sequence_length, sequence_length)
    # apply masking on upper triangle of matrix
    # For e.g. matrix of dimension 1, 3, 3
    #   [[[1. 0. 0.],
    #     [1. 1. 0.],
    #     [1. 1. 1.]]]
    mask = torch.tril(mask, 0)
    
    # convert to integers
    return mask.byte()

def mask(src, trg, pad_idx):
    """
        Apply masking to source caption input and target caption input for decoder.  
        Mask inputs where padding is applied so that padding is not used for attention.
        For source_caption, masking is only done for padded indices, hence all positions are used for attention
        For target_caption, masking is done for all future or subsequent positions to avoid looking into future caption words.
        Args:
            source_caption: source caption of shape (batch_size, sequence_length)
            target_caption: target caption of shape (batch_size, sequence_length)
        Returns:
            src_mask: source mask boolean tensor of shape (batch_size, 1, sequence_length) 
            trg_mask: target mask boolean tensor of shape (batch_size, 1, sequence_length) if provided.
    """
    # apply masking to padding index and make it 3D of shape (batch_size, 1, sequence_length)
    src_mask = (src != pad_idx).unsqueeze(1)
    
    if trg is not None:
        # apply masking to future indices and make it 3D of shape (batch_size, 1, sequence_length). change type of final mask to boolean tensor
        trg_mask = (trg != pad_idx).unsqueeze(-2) & subsequent_mask(trg.size(-1)).type_as(src_mask.data)

        return src_mask, trg_mask
    
    else:
        return src_mask

def clone(module, number_of_copies):
    """
        Create deep copy of module, which is used to stack same models one above other
        Args:
            module: Module to be copied
            number_of_copies: Number of copies to be made of module
        Returns:
            nn.ModuleList of size number_of_copies
    """
    return nn.ModuleList([deepcopy(module) for _ in range(number_of_copies)])

def attention(Q, K, V, mask):
    """
        Apply attention using Query(Q), Key(K), Value(V) and mask matrix.
        Suppose Transfore has dimension model_dimension then weight matrix for each Q, K, and V is transformed to model_dimension/heads in multi-head attention.
        So d_k = model_dimension/number_of_heads .
        All query, key and values has identical inputs then attention is known as self-attention

        Output is calculated by applying softmax to ( (Q * K') / sqrt(d_k) ) V

        Args:
            Q: Query matrix of shape (batch_size, number_of_heads, runtime_sequence_length, d_k)
            K: Query matrix of shape (batch_size, number_of_heads, runtime_sequence_length, d_k)
            V: Query matrix of shape (batch_size, number_of_heads, runtime_sequence_length, d_k)
            mask: Mask matrix to be applied of size (batch_size, 1, 1, sequence_length)
        
        Returns:
            Returns attention matrix of shape (batch_size, number_of_heads, runtime_sequence_length, d_k)
    """
    d_k = Q.size(-1) 
    QKt = Q.matmul(K.transpose(-1, -2)) 
    # The scaling is done to prevent the softmax function from being in the small gradient regions https://arxiv.org/pdf/1706.03762.pdf
    sm_input = QKt / np.sqrt(d_k)
    
    if mask is not None:
        #ISSUE: -inf should be avoided and instead using 1e-8
        sm_input = sm_input.masked_fill(mask == 0, -float(1e-8))
    
    softmax = F.softmax(sm_input, dim=-1)
    out = softmax.matmul(V)
    
    return out