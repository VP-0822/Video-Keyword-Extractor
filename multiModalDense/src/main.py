import os
from torch.utils import tensorboard as tensorboard
import torch
import numpy as np
from torch.utils.data import DataLoader

from util.util import averageOfAllMetrics
from util.fileUtil import savePytorchModel, loadPytorchModel
from dataHandler.multiModalDataIterator import MultiModalDataIterator
from model.transformer import MultiModalTransformer
from loss.lossComputer import LabelSmoothing, SimpleLossCompute
from scheduler.simpleScheduler import SimpleScheduler
from train import trainingLoop
from validate import validationLoop, evaluationLoopOnValidationSet

import config

try:
    import torch_xla.core.xla_model as xm
except ImportError:
    print('[WARNING]: Not able to import "torch_xla.core.xla_model", try pip install..')

def main():
    # prepare runtime
    os.makedirs(config.LOG_PATH, exist_ok=True)
    os.makedirs(config.EXPERIMENT_CHECKPOINT_FOLDER, exist_ok=True) # handles the case when model_checkpoint_path = LOG_PATH
    summary_writer = tensorboard.SummaryWriter(log_dir=config.LOG_PATH)
    summary_writer.add_text('config', 'Training for experiment name:' + config.CURRENT_EXPERIMENT_NAME, 0)
    summary_writer.add_text('config/comment', config.COMMENT, 0)
    print(f'Model log folder path: {config.LOG_PATH}')

    # initialize random seed
    torch.manual_seed(0)
    np.random.seed(0)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False
    #torch.cuda.set_device(cfg.device_ids[0])
    if config.USE_TPU:
        device = xm.xla_device()
        print('Using TPU device')
    else:
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(device)

    training_dataset = MultiModalDataIterator(config.VIDEO_HDF5_FILE_PATH, config.AUDIO_HDF5_FILE_PATH, device, config.TRAIN_META_FILE_PATH, \
            config.TRAIN_META_FILE_PATH, config.USE_CATEGORIES, config.USE_SUBTITLES, config.BATCH_SIZE, \
            min_word_occurance_freq=config.MINIMUM_WORD_OCCURANCE_FREQUENCY)

    validation_1_dataset = MultiModalDataIterator(config.VIDEO_HDF5_FILE_PATH, config.AUDIO_HDF5_FILE_PATH, device, config.VALIDATION_1_META_FILE_PATH, \
            config.TRAIN_META_FILE_PATH, config.USE_CATEGORIES, config.USE_SUBTITLES, config.BATCH_SIZE, \
            min_word_occurance_freq=config.MINIMUM_WORD_OCCURANCE_FREQUENCY)
    
    validation_2_dataset = MultiModalDataIterator(config.VIDEO_HDF5_FILE_PATH, config.AUDIO_HDF5_FILE_PATH, device, config.VALIDATION_2_META_FILE_PATH, \
            config.TRAIN_META_FILE_PATH, config.USE_CATEGORIES, config.USE_SUBTITLES, config.BATCH_SIZE, \
            min_word_occurance_freq=config.MINIMUM_WORD_OCCURANCE_FREQUENCY)

    val_1_maximum_caption_length = validation_1_dataset.getCaptionDataset().getPhaseMaximumCaptionLength()
    val_2_maximum_caption_length = validation_2_dataset.getCaptionDataset().getPhaseMaximumCaptionLength()

    # Create DataLoader for all datasets
    # DataLoader class stores provided training_dataset in .dataset property of DataLoader
    training_loader = DataLoader(training_dataset, collate_fn=training_dataset.dont_collate)
    validation_1_loader = DataLoader(validation_1_dataset, collate_fn=validation_1_dataset.dont_collate)
    validation_2_loader = DataLoader(validation_2_dataset, collate_fn=validation_2_dataset.dont_collate)

    # Create Transformer Model
    model = MultiModalTransformer(training_dataset.getCaptionDataset().getCaptionVocabSize(), training_dataset.getCaptionDataset().getSubtitleVocabSize(), \
            config.AUDIO_FEATURE_DIMENSION, config.VIDEO_FEATURE_DIMENSION, config.AUDIO_MODEL_DIMENSION, config.VIDEO_MODEL_DIMENSION, \
            config.SUBTITLE_MODEL_DIMENSION, config.AUDIO_FEEDFORWARD_UNITS, config.VIDEO_FEEDFORWARD_UNITS, config.SUBTITLE_FEEDFORWARD_UNITS, \
            config.AUDIO_ENCODER_DECODER_LAYERS, config.VIDEO_ENCODER_DECODER_LAYERS, config.SUBTITLE_ENCODER_DECODER_LAYERS, \
            config.DROPOUT_PERCENTAGE, config.NUMBER_OF_ATTENTION_HEADS, config.USE_LINEAR_EMBEDDER)
    
    # Create LabelSmoothing and Learning rate optimizer scheduler

    label_smoothing = LabelSmoothing(config.LABEL_SMOOTHING_FACTOR, training_dataset.getCaptionDataset().getPaddingTokenIndex())
    # lr = 0 here have no impact on training (see lr scheduler)
    optimizer = torch.optim.Adam(model.parameters(), 0, (config.ADAM_BETA_1, config.ADAM_BEAT_2), config.ADAM_EPS)
    lr_scheduler = SimpleScheduler(optimizer, config.LEARNING_RATE, device)
    loss_computer = SimpleLossCompute(label_smoothing, lr_scheduler)

    # transfer model to device
    model.to(device)

    # calculate number of parameters
    param_num = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'Trainable Model Parameter Size: {param_num}')
    summary_writer.add_scalar('debug/param_number', param_num, 0)

    # Start training and validation
    best_meteor_metrics = 0
    unchange_best_metric_counter = 0

    start_epoch = config.STARTING_EPOCH
    # If continuing training, load optimizer and model state dictionaries
    if config.CONTINUE_TRAINING:
        saved_dictionary = loadPytorchModel(config.CONTINUE_TRAINING_MODEL_FILE_PATH)
        start_epoch = int(saved_dictionary['epoch_number']) + 1
        model.load_state_dict(saved_dictionary['model_state_dict'])
        optimizer.load_state_dict(saved_dictionary['optimizer_state_dict'])
        best_meteor_metrics = saved_dictionary['best_meteor_metrics']

    for epoch in range(start_epoch, config.TOTAL_EPOCHS):
        unchange_best_metric_counter += 1

        if (unchange_best_metric_counter == config.EARLY_STOP_EPOCH_NUMBERS):
            print(f'Early stopping at epoch {epoch}: Since best metric model is unchanged for epochs: {unchange_best_metric_counter}')
            break
        
        trainingLoop(model, training_loader, loss_computer, lr_scheduler, epoch, summary_writer, config.USE_CATEGORIES)

        # Validate on both validation datasets
        validation_1_loss = validationLoop(model, validation_1_loader, loss_computer, lr_scheduler, epoch, val_1_maximum_caption_length, \
                                config.VIDEOS_TO_MONITOR, summary_writer, config.USE_CATEGORIES, '1')
        validation_2_loss = validationLoop(model, validation_2_loader, loss_computer, lr_scheduler, epoch, val_2_maximum_caption_length, \
                                config.VIDEOS_TO_MONITOR, summary_writer, config.USE_CATEGORIES, '2')

        avg_validation_loss = (validation_1_loss + validation_2_loss) / 2

        # Start predicting captions one word by word after certain epochs are done, to save time.
        # On these predictions, evaluation is done
        # if epoch >= config.EPOCH_NUMBER_TO_START_EVALUATION or (config.TOTAL_EPOCHS - 1 == epoch):
        if epoch >= config.EPOCH_NUMBER_TO_START_EVALUATION:
            # validating with ground truth proposals provided in ActivityNet
            validation_set_1_metrics = evaluationLoopOnValidationSet(model, validation_1_loader, epoch, val_1_maximum_caption_length, config.LOG_PATH, \
                [config.VALIDATION_1_REFERENCE_JSON], config.tIoUs_FOR_EVALUATION_METRIC, summary_writer, config.USE_CATEGORIES, '1')

            validation_set_2_metrics = evaluationLoopOnValidationSet(model, validation_2_loader, epoch, val_2_maximum_caption_length, config.LOG_PATH, \
                [config.VALIDATION_2_REFERENCE_JSON], config.tIoUs_FOR_EVALUATION_METRIC, summary_writer, config.USE_CATEGORIES, '2')
            
            val_1_avg_metrics = validation_set_1_metrics['Average across tIoUs']
            val_2_avg_metrics = validation_set_1_metrics['Average across tIoUs']

            average_metrics = averageOfAllMetrics(val_1_avg_metrics, val_2_avg_metrics)

            summary_writer.add_scalar('metrics/val_loss_avg', avg_validation_loss, epoch)
            summary_writer.add_scalar('metrics/meteor', average_metrics['METEOR'] * 100, epoch)
            summary_writer.add_scalar('metrics/bleu4', average_metrics['Bleu_4'] * 100, epoch)
            summary_writer.add_scalar('metrics/bleu3', average_metrics['Bleu_3'] * 100, epoch)
            summary_writer.add_scalar('metrics/bleu2', average_metrics['Bleu_2'] * 100, epoch)
            summary_writer.add_scalar('metrics/bleu1', average_metrics['Bleu_1'] * 100, epoch)
            summary_writer.add_scalar('metrics/rouge_l', average_metrics['ROUGE_L'] * 100, epoch)
            summary_writer.add_scalar('metrics/cider', average_metrics['CIDEr'] * 100, epoch)
            summary_writer.add_scalar('metrics/precision', average_metrics['Precision'] * 100, epoch)
            summary_writer.add_scalar('metrics/recall', average_metrics['Recall'] * 100, epoch)
            
            # Save model when meteor score is improved
            if best_meteor_metrics < average_metrics['METEOR']:
                best_meteor_metrics = average_metrics['METEOR']
                savePytorchModel(epoch, model, optimizer, avg_validation_loss, validation_set_1_metrics, validation_set_2_metrics, best_meteor_metrics, config.EXPERIMENT_CHECKPOINT_FOLDER)

        if config.SAVE_INTERMEDIATE and (epoch % config.SAVE_INTERMEDIATE_AT_EVERY_NTH_EPOCH == 0):
            savePytorchModel(epoch, model, optimizer, avg_validation_loss, None, None, best_meteor_metrics, config.EXPERIMENT_CHECKPOINT_FOLDER)

        if config.SAVE_MODEL_ON_LAST_EPOCH and (config.TOTAL_EPOCHS - 1 == epoch):
            savePytorchModel(epoch, model, optimizer, avg_validation_loss, None, None, best_meteor_metrics, config.EXPERIMENT_CHECKPOINT_FOLDER)

