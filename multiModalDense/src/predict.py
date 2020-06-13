import torch
import spacy
from model.modelUtil import mask

def greedyDecoder(model, final_features_for_model, maximum_caption_length, start_token_index, end_token_index, padding_token_index, categories=None):
    assert model.training == False, 'call model.eval first'
    
    # src_video (video features): (B, S, d_feat_vid) ex: [28, 11, 1024], 
    # src_audio (video features): (B, S, d_feat_aud) ex: [28, 11, 128],
    # src_subs (video features): (B, Ss, d_model_subs) ex: [28, 14, 512],
    src_video, src_audio, src_subs = final_features_for_model

    # a mask containing 1s if the ending tok occured, 0s otherwise
    # we are going to stop if ending token occured in every sequence
    # created to early stop if all samples in batch are predicted before maximum_caption_length
    completeness_mask = torch.zeros(len(src_video), 1).byte().to(src_video.device) # shape is (28, 1)

    with torch.no_grad():
        batch_size, S = src_audio.size(0), src_audio.size(1)
        # Create target_vocab_indices for all the items in batch, starting with start index, hence shape is (28, 1)
        target_vocab_indices = (torch.ones(batch_size, 1) * start_token_index).type_as(src_audio).long()

        while (target_vocab_indices.size(-1) <= maximum_caption_length) and (not completeness_mask.all()):
            src_mask, trg_mask = mask(src_video[:, :, 0], target_vocab_indices, padding_token_index)
            src_subs_mask = mask(src_subs, None, padding_token_index)
            masks = src_mask, trg_mask, src_subs_mask
            if categories is not None:
                preds = model(final_features_for_model, target_vocab_indices, masks, categories)
            else:
                preds = model(final_features_for_model, target_vocab_indices, masks)
            next_word = preds[:, -1].max(dim=-1)[1].unsqueeze(1) # Next word size (sequence_length, 1) where word index for each sequence length is predicted
            target_vocab_indices = torch.cat([target_vocab_indices, next_word], dim=-1) # Add next predicted word for all the batch samples to their respective target_vocab_indices row

            # sum two masks (or adding 1s where the ending token occured)
            completeness_mask = completeness_mask | torch.eq(next_word, end_token_index).byte()

    return target_vocab_indices

def predictForMonitorVideos(model, monitor_video_ids, validation_multimodal_dataset_iterator, maximum_caption_length, use_categories=None):
    """
        Predict caption for specified video one word by one word. Using greedy decoder functionality
    """
    video_metadata_list = validation_multimodal_dataset_iterator.getMultiModalDataset().getVideoMetaDataList()
    device = validation_multimodal_dataset_iterator.getDevice()
    start_token_index = validation_multimodal_dataset_iterator.getCaptionDataset().getStartTokenIndex()
    end_token_index = validation_multimodal_dataset_iterator.getCaptionDataset().getEndTokenIndex()
    padding_token_index = validation_multimodal_dataset_iterator.getCaptionDataset().getPaddingTokenIndex()
    training_subtitle_vocabs = validation_multimodal_dataset_iterator.getCaptionDataset().getTrainingSubtitleVocabs()
    training_caption_vocabs = validation_multimodal_dataset_iterator.getCaptionDataset().getTrainingCaptionVocabs()
    log_text = ''
    predicted_row_wise_caption_words_list = [] # Row-wise caption words list

    original_video_indecies = []
    original_start_time = []
    original_end_time = []
    for monitor_video_id in monitor_video_ids:
        meta_subset = video_metadata_list[video_metadata_list['video_id'] == monitor_video_id]
        # For each proposal subvideo 
        for (_, _, start, end, _, _, _, _, video_index) in meta_subset.values:
            original_video_indecies.append(video_index)
            original_start_time.append(start)
            original_end_time.append(end)

    filtered_video_ids, filter_video_start_times, filtered_video_end_times, filtered_video_duration_times, \
    filtered_video_categories, filtered_video_rgb_stacks, filtered_video_flow_stacks, filtered_audio_stacks = validation_multimodal_dataset_iterator.getMultiModalDataset().getItems(original_video_indecies)

    for iterator_index, video_index in enumerate(original_video_indecies):
        video_rgb_features = filtered_video_rgb_stacks[iterator_index]
        video_flow_features = filtered_video_flow_stacks[iterator_index]
        audio_features = filtered_audio_stacks[iterator_index]
        subtitle_features = encodeSubtitleToTokenIndices(training_subtitle_vocabs, video_index, video_metadata_list, start_token_index, end_token_index, device)
        
        video_id = filtered_video_ids[iterator_index]
        video_category = filtered_video_categories[iterator_index]
        video_start_time = original_start_time[iterator_index].cpu().numpy()[0]
        video_end_time = original_end_time[iterator_index].cpu().numpy()[0]

        log_text += f'\t {video_id} {video_index}\n'
        
        # need to make it 3D for decoder
        video_rgb_features = video_rgb_features.unsqueeze(0)
        video_flow_features = video_flow_features.unsqueeze(0)
        audio_features = audio_features.unsqueeze(0)

        # Video features for single video_index instance
        video_rgb_features = video_rgb_features.to(device)
        video_flow_features = video_flow_features.to(device)
        audio_features = audio_features.to(device)
        subtitle_features = subtitle_features.to(device)

        final_features_for_model = video_rgb_features + video_flow_features, audio_features, subtitle_features

        if use_categories:
            category = torch.tensor([video_category]).unsqueeze(0).to(device)
            target_vocab_indices = greedyDecoder(
                model, final_features_for_model, maximum_caption_length, start_token_index, end_token_index, padding_token_index, category)
        else:
            target_vocab_indices = greedyDecoder(
                model, final_features_for_model, maximum_caption_length, start_token_index, end_token_index, padding_token_index)
        
        target_vocab_indices = target_vocab_indices.cpu().numpy()[0]
        trg_words = [training_caption_vocabs.itos[i] for i in target_vocab_indices]
        predicted_row_wise_caption_words_list.append(trg_words)
        final_caption = ' '.join(trg_words)

        log_text += f'\t Predicted caption: {final_caption} \n'
        log_text += f'\t Ground truth proposals: {video_start_time//60:.0f}:{video_start_time%60:02.0f} {video_end_time//60:.0f}:{video_end_time%60:02.0f} \n'
        log_text += f'URL: https://www.youtube.com/embed/{video_id[2:]}?start={video_start_time}&end={video_end_time}&rel=0 \n'
        log_text += '\t \n'

    return log_text

def encodeSubtitleToTokenIndices(train_subs_vocab, video_index, video_metadata_list, start_token_index, end_token_index, device):
    subs = video_metadata_list.iloc[video_index]['subs']
    # check for 'nan'
    if subs != subs:
        subs = ''
    subs = [token.text for token in spacy.load('en').tokenizer(subs)]
    subs = [train_subs_vocab.stoi[word] for word in subs]
    subs = [start_token_index] + subs + [end_token_index]
    return torch.tensor(subs).unsqueeze(0).to(device)