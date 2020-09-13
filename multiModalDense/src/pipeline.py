from dataHandler.multiModalDataIterator import MultiModalDataIterator
import config
import torch
from torch.utils.data import DataLoader
from util.fileUtil import loadPytorchModel
from model.transformer import MultiModalTransformer
from predict import predictForMonitorVideos
from keywordExtractor.keywordExtractor import KeywordExtractor

def predictForSingleVideo(videoId):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    training_dataset = MultiModalDataIterator(config.VIDEO_HDF5_FILE_PATH, config.AUDIO_HDF5_FILE_PATH, device, config.TRAIN_META_FILE_PATH, \
            config.TRAIN_META_FILE_PATH, config.USE_CATEGORIES, config.USE_SUBTITLES, config.BATCH_SIZE, False, \
            min_word_occurance_freq=config.MINIMUM_WORD_OCCURANCE_FREQUENCY, video_mean_split=False, audio_mean_split=False)

    validation_2_dataset = MultiModalDataIterator(config.VIDEO_HDF5_FILE_PATH, config.AUDIO_HDF5_FILE_PATH, device, config.VALIDATION_2_META_FILE_PATH, \
            config.TRAIN_META_FILE_PATH, config.USE_CATEGORIES, config.USE_SUBTITLES, config.VALIDATION_BATCH_SIZE, False, \
            min_word_occurance_freq=config.MINIMUM_WORD_OCCURANCE_FREQUENCY, video_mean_split=False, audio_mean_split=False)
    
    val_2_maximum_caption_length = config.USE_DEFAULT_CAPTION_LENGTH
    validation_2_loader = DataLoader(validation_2_dataset, collate_fn=validation_2_dataset.dont_collate)
    dataset_iterator = validation_2_loader.dataset

    model = MultiModalTransformer(training_dataset.getCaptionDataset().getCaptionVocabSize(), training_dataset.getCaptionDataset().getSubtitleVocabSize(), \
            config.AUDIO_FEATURE_DIMENSION, config.VIDEO_FEATURE_DIMENSION, config.AUDIO_MODEL_DIMENSION, config.VIDEO_MODEL_DIMENSION, \
            config.SUBTITLE_MODEL_DIMENSION, config.AUDIO_FEEDFORWARD_UNITS, config.VIDEO_FEEDFORWARD_UNITS, config.SUBTITLE_FEEDFORWARD_UNITS, \
            config.AUDIO_ENCODER_DECODER_LAYERS, config.VIDEO_ENCODER_DECODER_LAYERS, config.SUBTITLE_ENCODER_DECODER_LAYERS, \
            config.DROPOUT_PERCENTAGE, config.NUMBER_OF_ATTENTION_HEADS, config.USE_LINEAR_EMBEDDER)

    model.to(device)

    # Update model values
    saved_dictionary = loadPytorchModel(config.CONTINUE_TRAINING_MODEL_FILE_PATH)
    model.load_state_dict(saved_dictionary['model_state_dict'])
    model.eval()

    _, predicted_dict = predictForMonitorVideos(model, [videoId], dataset_iterator, val_2_maximum_caption_length, to_json=True, trim_start_end=True)
    return predicted_dict

def predictKeywords(input_text, algo=config.KEYWORD_EXTRACTOR_ALGORITHM):
    ke = KeywordExtractor(algo)
    result_dict = ke.getKeywords(top_n=config.N_BEST_KEYWORDS, video_description=input_text)
    return result_dict

def populateKeywords(predicted_dict):
    for video_segment in predicted_dict['results']:
        # Keywords for caption
        video_segment['caption_keywords'] = predictKeywords(video_segment['caption'])
        # Keywords for caption + subtitles
        video_segment['caption_subtitle_keywords'] = predictKeywords(video_segment['caption'] + '.' + video_segment['subtitle'])
