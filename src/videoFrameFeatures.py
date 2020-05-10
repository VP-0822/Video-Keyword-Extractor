import pickle
import numpy as np
import random
import math
import os

class VideoFrameFeaturesExtractor:
    def __init__(self, videoFramePickleFilesList, maximumFrameLimit=40 ,no_samples=None , video_ids=None):
        self.no_samples = no_samples
        self.sample_video_ids = video_ids
        self.videoFramePickleFiles = videoFramePickleFilesList
        self.maximumFrameLimit = maximumFrameLimit
        if len(self.videoFramePickleFiles) == 1:
            self.mutlipartLoad = False
        else :
            self.mutlipartLoad = True
        self.loadVideoFrameFeatures()

    def _loadPickleFile(self):
        if self.mutlipartLoad is False:
            with open(self.videoFramePickleFiles[0], "rb") as fp:
                self.all_videos = pickle.load(fp)
                return
        final_video_dict = dict()
        for filePath in self.videoFramePickleFiles:
            with open(filePath, "rb") as fp:
                all_videos = pickle.load(fp)
                final_video_dict.update(all_videos)
        self.all_videos = final_video_dict

    def loadVideoFrameFeatures(self):
        self._loadPickleFile()
        if self.no_samples is None:
            return self.all_videos

        self.random_videos_frames = {}
        if self.sample_video_ids is not None:
            for video_id in self.sample_video_ids:
                if(len(self.all_videos[video_id]) > self.maximumFrameLimit):
                    random_frame_pick_interval = math.floor(len(self.all_videos[video_id]) / self.maximumFrameLimit)
                    self.random_videos_frames[video_id] = self.all_videos[video_id][0:random_frame_pick_interval*self.maximumFrameLimit:random_frame_pick_interval]
                else:
                    self.random_videos_frames[video_id] = self.all_videos[video_id]
            return
        all_video_ids = list(self.all_videos.keys())
        random.shuffle(all_video_ids)
        for index, video_id in enumerate(all_video_ids):
            if index is self.no_samples:
                break
            if(len(self.all_videos[video_id]) > self.maximumFrameLimit):
                random_frame_pick_interval = math.floor(len(self.all_videos[video_id]) / self.maximumFrameLimit)
                self.random_videos_frames[video_id] = self.all_videos[video_id][0:random_frame_pick_interval*self.maximumFrameLimit:random_frame_pick_interval]
            else:
                print('video frames count: ' + str(len(self.all_videos[video_id])))
                self.random_videos_frames[video_id] = self.all_videos[video_id]
        return

    def getVideoFrameFeatures(self):
        return self.random_videos_frames

if __name__ == "__main__":
    root_dir = os.path.dirname(__file__)
    PICKLE_FILE_PATH = os.path.join(root_dir,'../data/videoFeaturesPickle/all_video_frames.pkl')
    PICKLE_FILE_PART_1_PATH = os.path.join(root_dir,'../data/videoFeaturesPickle/all_video_frames_part1.pkl')
    PICKLE_FILE_PART_2_PATH = os.path.join(root_dir,'../data/videoFeaturesPickle/all_video_frames_part2.pkl')
    PICKLE_FILE_PART_3_PATH = os.path.join(root_dir,'../data/videoFeaturesPickle/all_video_frames_part3.pkl')
    
    # pickleFileList = [PICKLE_FILE_PATH]
    pickleFileList = [PICKLE_FILE_PART_1_PATH, PICKLE_FILE_PART_2_PATH, PICKLE_FILE_PART_3_PATH]
    vff = VideoFrameFeaturesExtractor(pickleFileList)
    sample_videos = vff.getVideoFrameFeatures()
    print(len(sample_videos.keys()))
    first_key = list(sample_videos.keys())[5]
    print(np.shape(sample_videos[first_key]))   