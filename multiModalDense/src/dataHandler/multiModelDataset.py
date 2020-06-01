from torch.utils.data.dataset import Dataset
import pandas as pd

class MultiModelDataset(Dataset):
    def __init__(self, video_feature_hdf5_file_path, audio_feature_hdf5_file_path, padding_token_index,  device, phase_meta_file_path, 
                preprocess_video_features=True, preprocess_audio_features=True, video_mean_split=True, audio_mean_split=True):
        self.video_feature_hdf5_file_path = video_feature_hdf5_file_path
        self.audio_feature_hdf5_file_path = audio_feature_hdf5_file_path
        self.padding_token_index = padding_token_index
        self.device = device
        self.phase_meta_file_path = phase_meta_file_path
        self.preprocess_video_features = preprocess_video_features
        self.preprocess_audio_features = preprocess_audio_features
        self.video_mean_split = video_mean_split
        self.audio_mean_split = audio_mean_split

        # Predefined constants
        self.VIDEO_FEATURE_DIM = 1024
        self.AUDIO_FEATURE_DIM = 128

        self._loadMetaFile()
    
    def _loadMetaFile(self):
        self.video_metadata_list = pd.read_csv(self.phase_meta_file_path, sep='\t')
    
    def __len__(self):
        return len(self.video_metadata_list)
    
    def __getitem__(self, indices):
        

    
