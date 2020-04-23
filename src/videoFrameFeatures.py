import pickle
import config
import numpy as np
import random
import math

MAX_FRAME_LIMIT = 60

def loadPickleFile(pickleFilePath):
    with open(pickleFilePath, "rb") as fp:
        all_videos = pickle.load(fp)
        return all_videos

def loadVideoFrameFeatures(pickleFilePath, numberOfVideos=None, video_ids=None):
    all_videos_frames = loadPickleFile(pickleFilePath)
    if numberOfVideos is None:
        return all_videos_frames

    random_videos_frames = {}
    if video_ids is not None:
        for video_id in video_ids:
            if(len(all_videos_frames[video_id]) > MAX_FRAME_LIMIT):
                random_frame_pick_interval = math.floor(len(all_videos_frames[video_id]) / MAX_FRAME_LIMIT)
                random_videos_frames[video_id] = all_videos_frames[video_id][0:random_frame_pick_interval*MAX_FRAME_LIMIT:random_frame_pick_interval]
            else:
                random_videos_frames[video_id] = all_videos_frames[video_id]
        return random_videos_frames
    all_video_ids = list(all_videos_frames.keys())
    random.shuffle(all_video_ids)
    for index, video_id in enumerate(all_video_ids):
        if index is numberOfVideos:
            break
        if(len(all_videos_frames[video_id]) > MAX_FRAME_LIMIT):
            random_frame_pick_interval = math.floor(len(all_videos_frames[video_id]) / MAX_FRAME_LIMIT)
            random_videos_frames[video_id] = all_videos_frames[video_id][0:random_frame_pick_interval*MAX_FRAME_LIMIT:random_frame_pick_interval]
        else:
            print('video frames count: ' + str(len(all_videos_frames[video_id])))
            random_videos_frames[video_id] = all_videos_frames[video_id]
    return random_videos_frames

if __name__ == "__main__":
    sample_videos = loadVideoFrameFeatures(config.PICKLE_FILE_PATH, 400)
    print(len(sample_videos.keys()))
    first_key = list(sample_videos.keys())[5]
    print(np.shape(sample_videos[first_key]))   