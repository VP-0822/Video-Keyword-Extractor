
import torch
import torch.nn as nn
import torch.nn.functional as F

"""
    Referred from http://nlp.seas.harvard.edu/2018/04/03/attention.html. Based on paper https://arxiv.org/pdf/1706.03762.pdf
"""

class LabelSmoothing(nn.Module):
    """
        The loss essentially drives your “gradients”, which in simple terms determines the “learning” of the model. 
        Many manual annotations are the results of multiple participants. They might have different criteria. They might make some mistakes. So complete reliance on correct label probabilites 
        for calculating loss is probably bad approach. To overcome we can apply label smoothing to relax the confidence for the labels.
        Label Smoothing formula to create expected one-hot vactors is as below
            new_onehot_labels = old_onehot_labels * (1 - label_smoothing) + label_smoothing / num_classes
    """
    def __init__(self, smoothing, padding_token_index):
        """
            Args:
                smoothing_factor: Smooting factor to be used in label smoothing
                padding_token_index: Padding token index
        """
        super(LabelSmoothing, self).__init__()
        self.smoothing = smoothing
        self.padding_token_index = padding_token_index
        
    def forward(self, predicted_tensor, target_tensor): # pred (B, S, V), target (B, S)
        """
            Apply label smoothing to obtained new loss for predicted tokens.
            Args:
                predicted_tensor: size (batch_size, target_sequence_length, caption_vocab_size)
                target_tensor: size (batch_size, target_sequence_length)
        """
        # Note: preds are expected to be after log
        batch_size, target_sequence_length, caption_vocab_size = predicted_tensor.shape
        # if batch_size =28, target_sequence_length = 20, caption_vocab_size= 1000
        predicted_tensor = predicted_tensor.contiguous().view(-1, caption_vocab_size) # shape (batch_size * target_sequence_length, caption_vocab_size) => (560, 1000)
        target_tensor = target_tensor.contiguous().view(-1) # shape (batch_size * target_sequence_length) => (560,) 1D tensor
        
        dist = self.smoothing * torch.ones_like(predicted_tensor) / (caption_vocab_size - 2) # -2 for start and end token
        # add smoothed ground-truth to prior (args: dim, index, src (value))
        dist.scatter_(1, target_tensor.unsqueeze(-1).long(), 1-self.smoothing) # pass 1-self.smooting for new_onehot_encooding
        # make the padding token to have zero probability
        dist[:, self.padding_token_index] = 0
        # ?? mask: 1 if target == pad_idx; 0 otherwise
        mask = torch.nonzero(target_tensor == self.padding_token_index)
        
        if mask.sum() > 0 and len(mask) > 0:
            # dim, index, val
            dist.index_fill_(0, mask.squeeze(), 0)
        
        # Apply Kullback-Leibler divergence Loss https://pytorch.org/docs/master/nn.functional.html#torch.nn.functional.kl_div
        return F.kl_div(predicted_tensor, dist, reduction='sum')
    
class SimpleLossCompute(object):
    """
        Simple loss compute object, which also takes step() for leraning rate scheduler.
    """
    
    def __init__(self, label_smoothing, lr_scheduler): 
        """
            Args:
                label_smoothing: label smoothing module.
                lr_scheduler: Learning rate scheduler
        """
        self.label_smoothing = label_smoothing
        self.lr_scheduler = lr_scheduler
        
    def __call__(self, predicted_tensor, target_tensor, number_of_tokens_in_actual_caption):
        """
            Calculate loss using label_smoothing criterion, do back propogation and upgrade gradient using learning rate scheduler.
            Args:
                predicted_tensor: size (batch_size, target_sequence_length, caption_vocab_size)
                target_tensor: size (batch_size, target_sequence_length)
                number_of_tokens_in_actual_caption: Number of tokens in actual caption to be used to normalise loss over all tokens
        """
        loss = self.label_smoothing(predicted_tensor, target_tensor) / number_of_tokens_in_actual_caption
        loss.backward()
    
        if self.lr_scheduler is not None:
            self.lr_scheduler.step()
            self.lr_scheduler.optimizer.zero_grad()
        
        return loss * number_of_tokens_in_actual_caption
