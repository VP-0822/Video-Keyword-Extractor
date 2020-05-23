import math
import os
import random
import fileUtils
from videoFrameFeatures import VideoFrameFeaturesExtractor
from youtube2TextCaptions import Youtube2TextCaptions
from captionPreprocess import CaptionPreprocessor
from gloveEmbeddings import GloveEmbedding

class DatasetPreprocessor:
    def __init__(self, no_samples, train_test_split, no_validation_samples, video_ids=None):
        self.total_samples_count = no_samples
        self.train_test_split = train_test_split
        self.number_of_validation_samples = no_validation_samples
        self.sample_video_ids = video_ids
    
    def loadVideoFeatureSamples(self, videoFramePickleFilesList):
        vff = VideoFrameFeaturesExtractor(videoFramePickleFilesList, no_samples=self.total_samples_count, video_ids=self.sample_video_ids)
        self.video_frames = vff.getVideoFrameFeatures()
        print('video frame features loaded')

    def splitIntoTrainTestAndValidation(self, trainVidIdFile=None, testVidIdFile=None, validationVidIdFile=None):
        train_test_split = math.floor((self.total_samples_count - self.number_of_validation_samples) * self.train_test_split)
        self.train_samples = dict()
        self.val_samples = dict()
        self.test_samples = dict()
        # Create fresh distribution into 3 sets
        if not os.path.exists(trainVidIdFile):
            test_size_counter = 0
            train_size_counter = 0
            for key in self.video_frames.keys():
                if (test_size_counter + train_size_counter) == (self.total_samples_count - self.number_of_validation_samples):
                    self.val_samples[key] = [self.video_frames[key]]
                    continue
                if(test_size_counter < train_test_split):
                    self.test_samples[key] = [self.video_frames[key]]
                    test_size_counter += 1
                    continue
                self.train_samples[key] = [self.video_frames[key]]
                train_size_counter += 1
            fileUtils.writeArrayToFile(trainVidIdFile, list(self.train_samples.keys()))
            fileUtils.writeArrayToFile(validationVidIdFile, list(self.val_samples.keys()))
            fileUtils.writeArrayToFile(testVidIdFile, list(self.test_samples.keys()))
        # load predefined sample distribution
        else:
            training_video_id_list = fileUtils.readArrayFromFile(trainVidIdFile)
            validation_video_id_list = fileUtils.readArrayFromFile(validationVidIdFile)
            testing_video_id_list = fileUtils.readArrayFromFile(testVidIdFile)
            for key in self.video_frames.keys():
                if key in training_video_id_list:
                    self.train_samples[key] = [self.video_frames[key]]
                if key in validation_video_id_list:
                    self.val_samples[key] = [self.video_frames[key]]
                if key in testing_video_id_list:
                    self.test_samples[key] = [self.video_frames[key]]
        
        print('Number of Training videos: ' + str(len(self.train_samples)))
        print('Number of Validation videos: ' + str(len(self.val_samples)))
        print('Number of Testing videos: ' + str(len(self.test_samples)))
    
    def attachInputCaptionsToVideos(self, captionSourceFile, embeddingSourceFile, embeddingDimension, captionsPerVideo, word_freq_threshold=2):
        final_video_ids = list(self.train_samples.keys())
        final_video_ids.extend(list(self.val_samples.keys()))

        # Load all captions for training and validation set
        y2tc = Youtube2TextCaptions(captionSourceFile, final_video_ids, captionsPerVideo)
        video_captions = y2tc.getFilteredCaptions()
        print('video captions loaded')
        self.caption_preprocessor = CaptionPreprocessor(video_captions,word_freq_threshold=word_freq_threshold)
        print('Final word count: ' + str(self.caption_preprocessor.getVocabSize()))
        #print(caption_preprocessor.getCaptionsVocabList())
        print('video captions preprocessed')
        glove_embedding = GloveEmbedding(embeddingSourceFile, embeddingDimension)
        print('glove embedding loaded')
        # Load vocabulary embedding vectors for training and validation vocabs
        self.vocab_word_embeddings = glove_embedding.getEmbeddingVectorFor(self.caption_preprocessor.getCaptionsVocabList(), self.caption_preprocessor.getVocabSize())
        preprocessed_video_captions = self.caption_preprocessor.caption_inputs

        for key in self.train_samples.keys():
            self.train_samples[key].append(preprocessed_video_captions[key])
        for key in self.val_samples.keys():
            self.val_samples[key].append(preprocessed_video_captions[key])

        # attach testing captions for reference
        y2tcForTest = Youtube2TextCaptions(captionSourceFile, list(self.test_samples.keys()), caption_per_video_limit=1)
        test_video_captions = y2tcForTest.getFilteredCaptions()
        for key, value in test_video_captions.items():
            self.test_samples[key].append(value)

    # This method returns expanded train and validation set across video per one caption
    def expandTrainAndValidationSet(self, trainingOrderVideoFile, shuffle_dataset=True):
        final_train_samples = dict()
        for key, value in self.train_samples.items():
            # Value is list with index 0 as frame_inputs and index 1 as all_video_captions
            for index, caption in enumerate(value[1]):
                sample_key = key + '^' + str(index + 1)
                final_train_samples[sample_key] = [value[0], caption]
        train_keys = list(final_train_samples.keys())
        if shuffle_dataset is True:
            random.shuffle(train_keys)
        self.shuffled_train_samples = dict()
        for key in train_keys:
            self.shuffled_train_samples[key] = final_train_samples[key]
        
        final_val_samples = dict()
        for key, value in self.val_samples.items():
            # Value is list with index 0 as frame_inputs and index 1 as all_video_captions
            for index, caption in enumerate(value[1]):
                sample_key = key + '^' + str(index + 1)
                final_val_samples[sample_key] = [value[0], caption]
        val_keys = list(final_val_samples.keys())
        if shuffle_dataset is True:
            random.shuffle(val_keys)
        self.shuffled_val_samples = dict()
        for key in val_keys:
            self.shuffled_val_samples[key] = final_val_samples[key]
        
        if os.path.exists(trainingOrderVideoFile):
            temp_train_samples = dict()
            temp_val_samples = dict()
            print('Following the previous training order')
            trainedVideoSampleIds = fileUtils.readArrayFromFile(trainingOrderVideoFile)
            for videoId in trainedVideoSampleIds:
                if videoId in self.shuffled_train_samples.keys():
                    temp_train_samples[videoId] = self.shuffled_train_samples[videoId]
                    continue
                if videoId in self.shuffled_val_samples.keys():
                    temp_val_samples[videoId] = self.shuffled_val_samples[videoId]
                    continue
            self.shuffled_train_samples = temp_train_samples
            self.shuffled_val_samples = temp_val_samples
        else:
            trainingSampleOrder = list()
            trainingSampleOrder.extend(list(self.shuffled_train_samples.keys()))
            trainingSampleOrder.extend(list(self.shuffled_val_samples.keys()))
            fileUtils.writeArrayToFile(trainingOrderVideoFile, trainingSampleOrder)
            print('Saved training sample order for continuing training')

        print('Expanded train samples: ' + str(len(self.shuffled_train_samples)))
        print('Expanded validation samples: ' + str(len(self.shuffled_val_samples)))
    
    def getTrainSamples(self):
        return self.shuffled_train_samples
    
    def getValidationSamples(self):
        return self.shuffled_val_samples
    
    def getTestSamples(self):
        return self.test_samples
    
    def getAllVideoIds(self):
        all_video_ids = list(self.test_samples.keys())
        all_video_ids.extend(list(self.train_samples.keys()))
        all_video_ids.extend(list(self.val_samples.keys()))  
        return all_video_ids
    
    def getCaptionPreprocessorInstance(self):
        return self.caption_preprocessor
    
    def getVocabWordEmbeddings(self):
        return self.vocab_word_embeddings
