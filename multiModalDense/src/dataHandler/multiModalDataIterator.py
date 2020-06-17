from torch.utils.data.dataset import Dataset
import dataHandler.captionDataset as cd
import dataHandler.multiModalDataset as mmd

class MultiModalDataIterator(Dataset):
    def __init__(self, video_feature_hdf5_file_path, audio_feature_hdf5_file_path, device, \
            phase_meta_file_path,  training_meta_file_path,use_yt_categories, use_asr_subtitles, batch_size, forOneByOnePrediction, \
            preprocess_video_features=True, preprocess_audio_features=True, video_mean_split=True, \
            audio_mean_split=True, split_size=4, min_word_occurance_freq=2):

        self.video_feature_hdf5_file_path = video_feature_hdf5_file_path
        self.audio_feature_hdf5_file_path = audio_feature_hdf5_file_path
        self.device = device
        self.phase_meta_file_path = phase_meta_file_path
        self.training_meta_file_path = training_meta_file_path
        self.use_yt_categories = use_yt_categories
        self.use_asr_subtitles = use_asr_subtitles
        self.batch_size = batch_size
        self.preprocess_video_features = preprocess_video_features
        self.preprocess_audio_features = preprocess_audio_features
        self.video_mean_split = video_mean_split
        self.audio_mean_split = audio_mean_split
        self.split_size = split_size
        self.min_word_occurance_freq = min_word_occurance_freq
        
        self.captionDataset = cd.CaptionDataset(cd.START_TOKEN, cd.END_TOKEN, \
            cd.PADDING_TOKEN, use_yt_categories, use_asr_subtitles, batch_size, device)
        self.captionDataset.createDataset(self.phase_meta_file_path, self.training_meta_file_path, self.min_word_occurance_freq)

        self.multiModalDataset = mmd.MultiModalDataset(video_feature_hdf5_file_path, \
            audio_feature_hdf5_file_path, self.captionDataset.getPaddingTokenIndex(),  device, phase_meta_file_path, forOneByOnePrediction, \
            preprocess_video_features, preprocess_audio_features, video_mean_split, audio_mean_split, split_size)

    def __getitem__(self, unused_dataset_index):
        captionData = self.captionDataset.getNextBatchItems()
        # All indices of batch items, this gives list of size of batch_size
        indices = captionData.idx
        multimodalData = self.multiModalDataset[indices]
        # actually returns all captionData items and multimodalData as comma seperated
        return_data = captionData, *multimodalData
        return return_data

    def __len__(self):
        return len(self.captionDataset.getCaptionDataset())
    
    def update_iterator(self):
        # reset captionIterator as it is created using python default iter(..)
        self.captionDataset.resetCaptionIterator()
    
    def getCaptionDataset(self):
        return self.captionDataset
    
    def getMultiModalDataset(self):
        return self.multiModalDataset
    
    def getDevice(self):
        return self.device
    
    def dont_collate(self, batch):
        return batch[0]