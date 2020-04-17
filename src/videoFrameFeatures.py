import pickle
import config
import numpy as np
import random

def loadPickleFile(pickleFilePath):
    with open(pickleFilePath, "rb") as fp:
        all_videos = pickle.load(fp)
        return all_videos

def loadVideoFrameFeatures(pickleFilePath, numberOfVideos=None):
    all_videos_frames = loadPickleFile(pickleFilePath)
    if numberOfVideos is None:
        return all_videos_frames
    all_video_ids = list(all_videos_frames.keys())
    random.shuffle(all_video_ids)
    random_videos_frames = {}
    for index, video_id in enumerate(all_video_ids):
        if index is numberOfVideos:
            break
        random_videos_frames[video_id] = all_videos_frames[video_id]
    return random_videos_frames

if __name__ == "__main__":
    sample_videos = loadVideoFrameFeatures(config.PICKLE_FILE_PATH, 15)
    print(sample_videos.keys())
    first_key = list(sample_videos.keys())[0]
    print(np.shape(sample_videos[first_key]))   