import numpy as np
class AudioFeatureDataset:
    def __init__(self, hdf5AudioFeatureInstance):
        self.audioFeatureFileInstance = hdf5AudioFeatureInstance
    
    def getAudioFeatures(self, videoIds):
        audioFeatures = list() 
        for _videoId in videoIds:
            _audioFeature = self.audioFeatureFileInstance.get(f'{_videoId}/vggish_features')
            if _audioFeature is None:
                _audioFeature = np.empty([0, 128], dtype=float)
            audioFeatures.append(_audioFeature)
        
        return np.array(audioFeatures)

if __name__ == "__main__":
    pass
