import glob
import numpy as np
import os
from tqdm import tqdm
import pickle

root_dir = os.path.dirname(__file__)
NPZ_VIDEO_FEATURE_FILE = os.path.join(root_dir, '../data/videoFeatures')
PICKLE_FILE_PATH = os.path.join(root_dir,'../data/videoFeaturesPickle/all_video_frames.pkl')
PICKLE_FILE_PART_1_PATH = os.path.join(root_dir,'../data/videoFeaturesPickle/all_video_frames_part1.pkl')
PICKLE_FILE_PART_2_PATH = os.path.join(root_dir,'../data/videoFeaturesPickle/all_video_frames_part2.pkl')
PICKLE_FILE_PART_3_PATH = os.path.join(root_dir,'../data/videoFeaturesPickle/all_video_frames_part3.pkl')

def convertToPickle(npzFilePaths, multipartSave=True):
    npz_files = glob.glob(os.path.join(npzFilePaths, '*.npz'))
    pretrained_video_features = dict()
    for npz_file in tqdm(npz_files):
        np_data = np.load(npz_file,allow_pickle=True)
        processed_video_ids = np_data['arr_1']
        all_frame_features = np_data['arr_0']
        for index, video_id in enumerate(processed_video_ids):
            pretrained_video_features[video_id] = all_frame_features[index]
    print('Total videos loaded: ' + str(len(pretrained_video_features)))
    if multipartSave is False:
        with open(PICKLE_FILE_PATH, "wb") as fp:
            pickle.dump(pretrained_video_features, fp)
        return
    pretrained_video_features_part1 = dict(list(pretrained_video_features.items())[:len(pretrained_video_features)//3])
    print('Part one size: ' + str(len(pretrained_video_features_part1)))
    pretrained_video_features_rest = dict(list(pretrained_video_features.items())[len(pretrained_video_features)//3:])
    pretrained_video_features_part2 = dict(list(pretrained_video_features_rest.items())[:len(pretrained_video_features_rest)//2])
    print('Part two size: ' + str(len(pretrained_video_features_part2)))
    pretrained_video_features_part3 = dict(list(pretrained_video_features_rest.items())[len(pretrained_video_features_rest)//2:])
    print('Part two size: ' + str(len(pretrained_video_features_part3)))

    with open(PICKLE_FILE_PART_1_PATH, "wb") as fp:
        pickle.dump(pretrained_video_features_part1, fp)
    with open(PICKLE_FILE_PART_2_PATH, "wb") as fp:
        pickle.dump(pretrained_video_features_part2, fp)
    with open(PICKLE_FILE_PART_3_PATH, "wb") as fp:
        pickle.dump(pretrained_video_features_part3, fp)
        
if __name__ == '__main__':
    convertToPickle(NPZ_VIDEO_FEATURE_FILE)