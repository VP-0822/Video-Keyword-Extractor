import h5py
import os
import numpy as np

def readHDF5FileInstance(hdf5FilePath):
    return h5py.File(hdf5FilePath, "r")
 

if __name__ == "__main__":
    root_dir = os.path.dirname(__file__)
    VIDEO_HDF5_FILE_PATH = 'C:/OneDrive - SRH IT/Video_Keyword_Extractor/data/ActivityNet-available/sub_activitynet_v1-3.i3d_25fps_stack24step24_2stream.hdf5'
    # VIDEO_HDF5_FILE_PATH = os.path.join(root_dir,'../data/features/sub_activitynet_v1-3.i3d_25fps_stack24step24_2stream.hdf5')
    videoFeatureData = readHDF5FileInstance(VIDEO_HDF5_FILE_PATH)
    AUDIO_HDF5_FILE_PATH = 'C:/OneDrive - SRH IT/Video_Keyword_Extractor/data/ActivityNet-available/sub_activitynet_v1-3.vggish.hdf5'
    # AUDIO_HDF5_FILE_PATH = os.path.join(root_dir,'../data/features/sub_activitynet_v1-3.vggish.hdf5')
    audioFeatureData = readHDF5FileInstance(AUDIO_HDF5_FILE_PATH)
    sampleVideoKey = list(videoFeatureData.keys())[10]
    videoKeyData = list(videoFeatureData.get(f'{sampleVideoKey}'))
    videoRGBData = list(videoFeatureData.get(f'{sampleVideoKey}/i3d_features/rgb'))
    videoFlowData = list(videoFeatureData.get(f'{sampleVideoKey}/i3d_features/flow'))
    videoInfo = list(videoFeatureData.get(f'{sampleVideoKey}/i3d_info'))
    # videoC3DData = list(videoFeatureData.get(f'{sampleVideoKey}/c3d_features'))
    audioKeyData = list(audioFeatureData.get(f'{sampleVideoKey}'))
    audioData = list(audioFeatureData.get(f'{sampleVideoKey}/vggish_features'))
    audioInfo = list(audioFeatureData.get(f'{sampleVideoKey}/vggish_info'))
    print(videoKeyData)
    # print(videoRGBData[0])
    np_videoRGBData = np.array(videoRGBData)
    print('Video RGB shape')
    print(np_videoRGBData.shape)
    # print(videoFlowData[0])
    np_videoFlowData = np.array(videoFlowData)
    print('Video Flow shape')
    print(np_videoFlowData.shape)
    # print(videoInfo)
    # print(audioKeyData)
    # print(audioData[0])
    np_audioData = np.array(audioData)
    print('Video Audio shape')
    print(np_audioData.shape)
    # print(audioInfo)
    # print(videoC3DData)