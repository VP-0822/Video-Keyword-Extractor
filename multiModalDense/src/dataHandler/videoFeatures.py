import numpy as np
class VideoFeatureDataset:
    def __init__(self, hdf5VideoFeatureInstance):
        self.videoFeatureFileInstance = hdf5VideoFeatureInstance
    
    def getRGBFeatures(self, videoIds):
        videoFeatures = list() 
        for _videoId in videoIds:
            _videoRGBFeature = self.videoFeatureFileInstance.get(f'{_videoId}/i3d_features/rgb')
            videoFeatures.append(_videoRGBFeature)
        
        return np.array(videoFeatures)
    
    def getFlowFeatures(self, videoIds):
        videoFeatures = list() 
        for _videoId in videoIds:
            _videoFlowFeature = self.videoFeatureFileInstance.get(f'{_videoId}/i3d_features/flow')
            videoFeatures.append(_videoFlowFeature)
        
        return np.array(videoFeatures)

if __name__ == "__main__":
    pass
