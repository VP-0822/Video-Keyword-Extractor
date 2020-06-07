from tqdm import tqdm
from time import time, strftime, localtime
from model.modelUtil import mask
from torch.utils.data import DataLoader
import numpy as np
import torch

def trainingLoop(model, dataset_loader, loss_computer, lr_scheduler, epoch_number, summary_writer, use_categories):
    """
        Training loop for training the entire training data for transformer model. Following process done,
        1. Reset dataset iterator for each new epoch
        2. enumerate dataset iterator batch wise
            a. Create mask matrix for various inputs such as Subtitle, Audio, and Video
            b. Predict model
            c. Compute loss on prediction
            d. log computed losses in summary_writer
        Args:
            model: Training model object
            dataset_loader: The entire dataset loader instance of DataLoader
            loss_computer: Loss computer object
            lr_scheduler : Simple Learning Rate Schedular object
            epoch_number: Epoch number
            summary_writer: Tensorboard SummaryWriter object Read at https://pytorch.org/docs/stable/tensorboard.html
            use_categories: Use youtube category tag as input

    """
    model.train() # Just sets model to training mode, hence effective activates droupout and batch-normalization layer
    losses = []
    
    dataset_loader.dataset.update_iterator() # Call update iterator to reset caption iterator in MultiModelDataIterator.
    
    time = strftime('%X', localtime()) # Current time in HH:MM:SS 

    dataset_iterator = dataset_loader.dataset

    # Iterate over dataset
    for i, batch_data in enumerate(tqdm(dataset_loader, desc=f'{time} training epoch: ({epoch_number})')):
        
        batch_loss_normalised = computeLossForABatch(model, 'train', batch_data, dataset_iterator, use_categories, loss_computer)
        losses.append(batch_loss_normalised.item())

        if summary_writer is not None:
            step_num = epoch_number * len(dataset_loader) + i
            summary_writer.add_scalar('train/Loss_iter', batch_loss_normalised.item(), step_num)
            summary_writer.add_scalar('debug/lr', lr_scheduler.get_lr(), step_num)
    
    # we have already divided it
    epoch_loss = np.sum(losses) / len(dataset_loader)
    
    if summary_writer is not None:
        summary_writer.add_scalar('debug/train_loss_epoch', epoch_loss, epoch_number)

def computeLossForABatch(model, phase, batch_data, dataset_iterator, use_categories, loss_computer):
    """
        Args:
            model: Training model object
            phase: Phase of model learning 'train' and 'validate'
            batch_data: data for the batch
            dataset_iterator: Instance of MultiModelDataIterator
            loss_computer: Loss computer object
            use_categories: Use youtube category tag as input
        Returns:
            batch_loss_norm: returns normalized batch loss 
    """
    caption_data, _, _, _, _, video_categories, video_rgb_feature_stack, video_flow_feature_stack, audio_feature_stack = batch_data

    captions = caption_data.caption # All batch video captions by index
    input_captions = captions[:, :-1] # Remove last caption token 'end of statement token'
    actual_captions = captions[:, :-1] # Remove first caption token 'start token'. Size (batch_size, target_sequence_length)
    subtitle_stack = caption_data.subs
    feature_stacks = video_rgb_feature_stack + video_flow_feature_stack, audio_feature_stack, subtitle_stack

    # video_rgb_feature_stack : shape (batch_size, sequence_length, video_feature_dimension) For e.g. (28, 40, 1024)
    # video_flow_feature_stack : shape (batch_size, sequence_length, video_feature_dimension) For e.g. (28, 40, 1024)
    # audio_feature_stack : shape (batch_size, sequence_length, video_feature_dimension) For e.g. (28, 40, 128)
    # subtitle_stack : shape (batch_size, sequence_length) For e.g. (28, 200)
    
    # For video and audio create only one mask matrix by eliminating last dimension to make mask matrix shape (batch_size, sequence_length)
    audio_video_common_masks = mask(feature_stacks[0][:, :, 0], input_captions, dataset_iterator.getCaptionDataset().getPaddingTokenIndex())
    # Create subtitle mask
    subs_mask = mask(feature_stacks[-1], None, dataset_iterator.getCaptionDataset().getPaddingTokenIndex())
    masks = *audio_video_common_masks, subs_mask
    # Final mask above contains masks for (audio/video), target_caption_mask, subtitle_mask
    
    number_of_tokens_in_actual_captions = (actual_captions != dataset_iterator.getCaptionDataset().getPaddingTokenIndex()).sum()
    if 'validate' in phase:
        with torch.no_grad():
            if use_categories:
                # Predicted tensor size is (batch_size, target_sequence_length, caption_vocab_size)
                pred = model(
                    feature_stacks, input_captions, masks, video_categories
                )
            else:
                pred = model(feature_stacks, input_captions, masks)
            
            batch_loss = loss_computer(pred, actual_captions, number_of_tokens_in_actual_captions)
            # create normalized loss
            batch_loss_norm = batch_loss / number_of_tokens_in_actual_captions
            return batch_loss_norm
    else:
        if use_categories:
            # Predicted tensor size is (batch_size, target_sequence_length, caption_vocab_size)
            pred = model(
                feature_stacks, input_captions, masks, video_categories
            )
        else:
            pred = model(feature_stacks, input_captions, masks)
        
        batch_loss = loss_computer(pred, actual_captions, number_of_tokens_in_actual_captions)
        # create normalized loss
        batch_loss_norm = batch_loss / number_of_tokens_in_actual_captions
        return batch_loss_norm
