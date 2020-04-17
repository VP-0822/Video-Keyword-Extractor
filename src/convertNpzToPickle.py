import glob
import numpy as np
import os
from tqdm import tqdm
import config
import pickle

def convertToPickle(npzFilePaths):
    npz_files = glob.glob(os.path.join(npzFilePaths, '*.npz'))
    pretrained_video_features = dict()
    for npz_file in tqdm(npz_files):
        np_data = np.load(npz_file,allow_pickle=True)
        processed_video_ids = np_data['arr_1']
        all_frame_features = np_data['arr_0']
        for index, video_id in enumerate(processed_video_ids):
            pretrained_video_features[video_id] = all_frame_features[index]
    print('Total videos loaded: ' + str(len(pretrained_video_features)))
    with open(config.PICKLE_FILE_PATH, "wb") as fp:
        pickle.dump(pretrained_video_features, fp)
        
if __name__ == '__main__':
    convertToPickle(config.NPZ_VIDEO_FEATURE_FILE)