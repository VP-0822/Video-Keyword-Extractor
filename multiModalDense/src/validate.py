from evaluate.metric import calculateMetrics
from predict import predictForMonitorVideos, greedyDecoder
import json
from time import time, strftime, localtime
from tqdm import tqdm
import os
import numpy as np
from train import computeLossForABatch

def validationLoop(model, dataset_loader, loss_computer, lr_scheduler, epoch_number, maximum_caption_length, videos_to_monitor, summary_writer, use_categories, validation_set_no):
    """
        Validation on specified validatation dataset for transformer architecture. Following process is performed:
        1. Reset dataset iterator for each new epoch
        2. enumerate dataset iterator batch wise
            
    """
    validation_set_name = 'validate_' +  validation_set_no
    model.eval()
    losses = []
        
    dataset_loader.dataset.update_iterator()
    time = strftime('%X', localtime())
    dataset_iterator = dataset_loader.dataset

    for i, batch_data in enumerate(tqdm(dataset_loader, desc=f'{time} {validation_set_name} ({epoch_number})')):
        batch_loss_normalised = computeLossForABatch(model, validation_set_name, batch_data, dataset_iterator, use_categories, loss_computer)
        
        losses.append(batch_loss_normalised.item())
        print('Validation batch loss:')
        print(batch_loss_normalised.item())
        if summary_writer is not None:
            step_num = epoch_number * len(dataset_loader) + i
            summary_writer.add_scalar(f'debug/{validation_set_name}_loss_iter', batch_loss_normalised.item(), step_num)

    epoch_loss = np.sum(losses) / len(dataset_loader)

    print('Validation epoch loss:')
    print(epoch_loss)

    if summary_writer is not None:
        summary_writer.add_scalar(f'debug/{validation_set_name}_loss_epoch', epoch_loss, epoch_number)
        
        if validation_set_name == 'validate_1':
            if use_categories:
                log_text = predictForMonitorVideos(model, videos_to_monitor, dataset_iterator, maximum_caption_length, use_categories)
            else:
                log_text = predictForMonitorVideos(model, videos_to_monitor, dataset_iterator, maximum_caption_length)
            summary_writer.add_text(f'prediction_1by1_{validation_set_name}', log_text, epoch_number)
        
    return epoch_loss

def evaluationLoopOnValidationSet(model, dataset_loader, epoch, maximum_caption_length, log_path, reference_paths, tIoUs, \
            summary_writer, use_categories, validation_set_no):
    validation_set_name = 'validate_' +  validation_set_no
    start_timer = time()
    time_ = strftime('%X', localtime())
    
    # init the dict with results and other technical info
    predictions = {
        'version': 'VERSION 1.0',
        'external_data': {
            'used': True, 
            'details': ''
        },
        'results': {}
    }
    model.eval()
    dataset_loader.dataset.update_iterator()
    multimodal_dataset_iterator = dataset_loader.dataset
    
    start_token_index = multimodal_dataset_iterator.getCaptionDataset().getStartTokenIndex()
    end_token_index = multimodal_dataset_iterator.getCaptionDataset().getEndTokenIndex()
    pad_token_index = multimodal_dataset_iterator.getCaptionDataset().getPaddingTokenIndex()
    training_caption_vocabs = multimodal_dataset_iterator.getCaptionDataset().getTrainingCaptionVocabs()

    tqdm_title = f'{time_} 1-by-1 gt proposals ({epoch})'
    
    for i, batch_data in enumerate(tqdm(dataset_loader, desc=tqdm_title)):
        caption_data, video_ids, _, starts, ends, video_categories, video_rgb_feature_stack, video_flow_feature_stack, audio_feature_stack = batch_data
        feature_stacks = video_rgb_feature_stack + video_flow_feature_stack, audio_feature_stack, caption_data.subs
        
        ### PREDICT TOKENS ONE-BY-ONE AND TRANSFORM THEM INTO STRINGS TO FORM A SENTENCE
        if use_categories:
            target_vocab_indices = greedyDecoder(model, feature_stacks, maximum_caption_length, start_token_index, end_token_index, pad_token_index, video_categories)
        else:
            target_vocab_indices = greedyDecoder(model, feature_stacks, maximum_caption_length, start_token_index, end_token_index, pad_token_index)
        target_vocab_indices = target_vocab_indices.cpu().numpy() # what happens here if I use only cpu?
        # transform integers into strings
        predicted_row_wise_caption_tokens_list = [[training_caption_vocabs.itos[index] for index in indices] for indices in target_vocab_indices]

        # initialize the list to fill it using indices instead of appending them
        final_caption_list = [None] * len(predicted_row_wise_caption_tokens_list) # batch size

        for iterator_index, tokens in enumerate(predicted_row_wise_caption_tokens_list):
            # remove starting token
            tokens = tokens[1:]
            # and remove everything after ending token
            # sometimes it is not in the list
            try:
                first_entry_of_eos = tokens.index('</s>')
                tokens = tokens[:first_entry_of_eos]
            except ValueError:
                pass
            # remove the period at the eos, if it is at the end (safe)
            # if trg_strings[-1] == '.':
            #     trg_strings = trg_strings[:-1]
            # join everything together
            caption = ' '.join(tokens)
            # Capitalize the sentence
            caption = caption.capitalize()
            # add the filtered sentense to the list
            final_caption_list[iterator_index] = caption
            
        # Add caption predictions for each video to prediction dictonary
        for video_id, start, end, sentence in zip(video_ids, starts, ends, final_caption_list):
            segment = {
                'sentence': sentence,
                'timestamp': [start.item(), end.item()]
            }

            if predictions['results'].get(video_id):
                predictions['results'][video_id].append(segment)
            else:
                predictions['results'][video_id] = [segment]
    
    if log_path is None:
        return None
    else:
        # Save JSON submission file
        save_filename = f'predictions_{validation_set_name}_epoch{epoch}.json'
        submission_path = os.path.join(log_path, save_filename)

        # in case summary_writer is not defined make logdir
        os.makedirs(log_path, exist_ok=True)

        # make backup of previous run for same validation set
        if os.path.exists(submission_path):
            submission_path = submission_path.replace('.json', f'_{time()}.json')

        with open(submission_path, 'w') as outf:
            json.dump(predictions, outf)

        val_metrics = calculateMetrics(reference_paths, submission_path, tIoUs)

        if (summary_writer is not None):
            summary_writer.add_scalar(f'{validation_set_name}/meteor', val_metrics['Average across tIoUs']['METEOR'] * 100, epoch)
            summary_writer.add_scalar(f'{validation_set_name}/bleu4', val_metrics['Average across tIoUs']['Bleu_4'] * 100, epoch)
            summary_writer.add_scalar(f'{validation_set_name}/bleu3', val_metrics['Average across tIoUs']['Bleu_3'] * 100, epoch)
            summary_writer.add_scalar(f'{validation_set_name}/bleu2', val_metrics['Average across tIoUs']['Bleu_2'] * 100, epoch)
            summary_writer.add_scalar(f'{validation_set_name}/bleu1', val_metrics['Average across tIoUs']['Bleu_1'] * 100, epoch)
            summary_writer.add_scalar(f'{validation_set_name}/rouge_l', val_metrics['Average across tIoUs']['ROUGE_L'] * 100, epoch)
            summary_writer.add_scalar(f'{validation_set_name}/cider', val_metrics['Average across tIoUs']['CIDEr'] * 100, epoch)
            summary_writer.add_scalar(f'{validation_set_name}/precision', val_metrics['Average across tIoUs']['Precision'] * 100, epoch)
            summary_writer.add_scalar(f'{validation_set_name}/recall', val_metrics['Average across tIoUs']['Recall'] * 100, epoch)
            summary_writer.add_scalar(f'{validation_set_name}/duration_of_1by1', (time() - start_timer) / 60, epoch)

        return val_metrics