from torch.utils.data.dataset import Dataset
from dataHandler.videoFeatures import VideoFeatureDataset
from dataHandler.audioFeatures import AudioFeatureDataset
import pandas as pd
import h5py
import torch
from torch.nn.utils.rnn import pad_sequence

class MultiModalDataset(Dataset):
    def __init__(self, video_feature_hdf5_file_path, audio_feature_hdf5_file_path, padding_token_index,  device, phase_meta_file_path, forOneByOnePrediction, 
                preprocess_video_features=True, preprocess_audio_features=True, video_mean_split=True, audio_mean_split=True, split_size=4):
        self.video_feature_hdf5_file_path = video_feature_hdf5_file_path
        self.audio_feature_hdf5_file_path = audio_feature_hdf5_file_path
        self.padding_token_index = padding_token_index
        self.device = device
        self.phase_meta_file_path = phase_meta_file_path
        self.preprocess_video_features = preprocess_video_features
        self.preprocess_audio_features = preprocess_audio_features
        self.video_mean_split = video_mean_split
        self.audio_mean_split = audio_mean_split
        self.feature_split_size = split_size
        self.forOneByOnePrediction = forOneByOnePrediction

        # Predefined constants
        self.VIDEO_FEATURE_DIM = 1024
        self.AUDIO_FEATURE_DIM = 128

        self._loadMetaFile()
        self._loadMultiModalFeatures()
    
    def _loadMetaFile(self):
        self.video_metadata_list = pd.read_csv(self.phase_meta_file_path, sep='\t')
    
    def _loadMultiModalFeatures(self):
        self.video_hdf5_file_instance = h5py.File(self.video_feature_hdf5_file_path, "r")
        self.audio_hdf5_file_instance = h5py.File(self.audio_feature_hdf5_file_path, "r")
    
    def __len__(self):
        return len(self.video_metadata_list)
    
    def __getitem__(self, indices):
        return self.getItems(indices, self.forOneByOnePrediction)

    def getItems(self, indices, forOneByOnePrediction=False):
        video_ids, start_time_list, end_time_list, full_video_durations, categories = [], [], [], [], []

        # Iterate over batch items which are indexs of videos from meta file
        for index in indices:
            if type(index) is int:
                original_index = index
            else:
                original_index = index.item()
            video_id, caption, start_time, end_time, full_video_duration, category, _, _, _ = self.video_metadata_list.iloc[original_index]
            
            video_ids.append(video_id)
            start_time_list.append(start_time)
            end_time_list.append(end_time)
            categories.append(category)
            full_video_durations.append(full_video_duration)

        filtered_video_rgb_stacks, filtered_video_flow_stacks, filtered_audio_stacks = self.getFilteredFeatureStack(video_ids, start_time_list, end_time_list, full_video_durations, forOneByOnePrediction)

        # make other tensors 2D
        T_start_time = torch.tensor(start_time_list, device=self.device).unsqueeze(1)
        T_end_time = torch.tensor(end_time_list, device=self.device).unsqueeze(1)
        T_categories = torch.tensor(categories, device=self.device).unsqueeze(1)
        T_durations = torch.tensor(full_video_durations, device=self.device).unsqueeze(1)

        # video_rgb_features: shape (sequence_length, video_feature_dimension) For e.g. (225, 1024)
        # video_flow_features: shape (sequence_length, video_feature_dimension) For e.g. (225, 1024)
        # audio_rgb_features: shape (sequence_length, audio_feature_dimension) For e.g. (224, 128)
        return video_ids, T_start_time, T_end_time, T_durations, T_categories, filtered_video_rgb_stacks, filtered_video_flow_stacks, filtered_audio_stacks

    def getFilteredFeatureStack(self, video_ids, start_time_list, end_time_list, full_video_durations, forOneByOnePrediction):
        filtered_video_rgb_stacks, filtered_video_flow_stacks, filtered_audio_stacks = [], [], []

        # get video RGB and flow features from hdf5 file
        videoFeatureDataset = VideoFeatureDataset(self.video_hdf5_file_instance)
        # get audio features from hdf5 file
        audioFeatureDataset = AudioFeatureDataset(self.audio_hdf5_file_instance)

        raw_video_rgb_stacks = videoFeatureDataset.getRGBFeatures(video_ids)
        raw_video_flow_stacks = videoFeatureDataset.getFlowFeatures(video_ids)
        raw_audio_stacks = audioFeatureDataset.getAudioFeatures(video_ids)

        for index, videoId in enumerate(video_ids):
            start_time = start_time_list[index]
            end_time = end_time_list[index]
            full_video_length = full_video_durations[index]
            video_rgb_features = raw_video_rgb_stacks[index]
            video_flow_features = raw_video_flow_stacks[index]
            audio_features = raw_audio_stacks[index]

            T_video_rgb_features, T_video_flow_features, T_audio_features = self._filterSinglevideo(start_time, end_time, \
                                            full_video_length, video_rgb_features, video_flow_features, audio_features)
            
            # save tensors to device
            T_video_rgb_features = T_video_rgb_features.to(self.device)
            T_video_flow_features = T_video_flow_features.to(self.device)
            T_audio_features = T_audio_features.to(self.device)

            filtered_video_rgb_stacks.append(T_video_rgb_features)
            filtered_video_flow_stacks.append(T_video_flow_features)
            filtered_audio_stacks.append(T_audio_features)

        if forOneByOnePrediction is False:
            # apply padding to all features. padding 0 is applied to flow features as per literature referred [RGB and Flow are summed anyways]
            filtered_video_rgb_stacks = pad_sequence(filtered_video_rgb_stacks, batch_first=True, padding_value=self.padding_token_index)
            filtered_video_flow_stacks = pad_sequence(filtered_video_flow_stacks, batch_first=True, padding_value=0)
            filtered_audio_stacks = pad_sequence(filtered_audio_stacks, batch_first=True, padding_value=self.padding_token_index)

        return filtered_video_rgb_stacks, filtered_video_flow_stacks, filtered_audio_stacks

    def _filterSinglevideo(self, start_time, end_time, full_video_length, video_rgb_features, video_flow_features, audio_features):
        """
            Returns all modality features
            Returns:
                video_rgb_features: shape (sequence_length, video_feature_dimension) For e.g. (225, 1024)
                video_flow_features: shape (sequence_length, video_feature_dimension) For e.g. (225, 1024)
                audio_rgb_features: shape (sequence_length, audio_feature_dimension) For e.g. (224, 128)
        """
        assert video_rgb_features.shape == video_flow_features.shape
        video_timesteps, video_feature_dimension = video_rgb_features.shape
        audio_timesteps, audio_feature_dimension = audio_features.shape
        
        final_video_timesteps = 0
        # Since there might be few timesteps more or less in audio track, adjust features accordingly
        if video_timesteps > audio_timesteps:
            video_rgb_features = video_rgb_features[:audio_timesteps, :]
            video_flow_features = video_flow_features[:audio_timesteps, :]
            final_video_timesteps = audio_timesteps
        elif video_timesteps < audio_timesteps:
            audio_features = audio_features[:video_timesteps, :]
            final_video_timesteps = video_timesteps
        else:
            final_video_timesteps = video_timesteps
        
        # split features as per start_time and end_time, since considering segments of videos for captioning
        start_fraction = start_time / full_video_length
        end_fraction = end_time / full_video_length
        start_index = int(final_video_timesteps * start_fraction)
        end_index = int(final_video_timesteps * end_fraction)

        # handle scenario where segment is very small
        if start_index == end_index:
            print('Found very small video segment')
            # [T:T] -> [T-1:T] (T can be either T_video or T_audio)
            if start_index == final_video_timesteps:
                start_index -= 1
            # [T:T] -> [T:T+1]
            else:
                end_index += 1

        # trim to only segment features
        video_segment_rgb_features = video_rgb_features[start_index:end_index, :]
        video_segment_flow_features = video_flow_features[start_index:end_index, :]
        audio_segment_features = audio_features[start_index:end_index, :]

        # convert to pytorch tensors
        T_video_segment_rgb_features = torch.tensor(video_segment_rgb_features).float()
        T_video_segment_flow_features = torch.tensor(video_segment_flow_features).float()
        T_audio_segment_features = torch.tensor(audio_segment_features).float()
        
        if len(T_video_segment_rgb_features) == 0:
            print(f'Found video without audio or video input, initializing with random values')
            T_video_segment_rgb_features = torch.rand(1, self.VIDEO_FEATURE_DIM, device=self.device)
            T_video_segment_flow_features = torch.rand(1, self.VIDEO_FEATURE_DIM, device=self.device)
            T_audio_segment_features = torch.rand(1, self.AUDIO_FEATURE_DIM, device=self.device)

        # reduce timesteps by averaging over small groups
        if self.preprocess_video_features:
            T_video_segment_rgb_features = self.reduceFeatureTimesteps(T_video_segment_rgb_features, self.video_mean_split)
            T_video_segment_flow_features = self.reduceFeatureTimesteps(T_video_segment_flow_features, self.video_mean_split)
        if self.preprocess_audio_features:
            T_audio_segment_features = self.reduceFeatureTimesteps(T_audio_segment_features, self.audio_mean_split)

        return T_video_segment_rgb_features, T_video_segment_flow_features, T_audio_segment_features

    def reduceFeatureTimesteps(self, tensor, average_split, use_overlap=False):
        if len(tensor) == 0:
            return tensor
        
        if use_overlap:
            # remove overlapping features
            tensor = tensor[::2, :]

        if average_split:
            # split tensor into groups of size feature_split_size arrays, apply mean to each group on 0th dim to get 1D tensor
            tensor = [split_segment.mean(dim=0) for split_segment in torch.split(tensor, self.feature_split_size)]
            # stack all 1D tensors to form 2D tensor
            tensor = torch.stack(tensor)

        return tensor
    
    def getVideoMetaDataList(self):
        return self.video_metadata_list
    
