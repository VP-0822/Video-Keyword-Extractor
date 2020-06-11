import numpy as np
from torchtext import data
from torch.utils.data.dataset import Dataset

"""
    Uses torchtext to create TabularDataset. (Docs: https://pytorch.org/text/index.html)
"""

START_TOKEN = '<s>'
END_TOKEN = '</s>'
PADDING_TOKEN = '<empty>'
UNKNOWN_TOKEN = '<unk>' # Issue in pytorch need to use this if we need min_freq > 1
    
class CaptionDataset():
    def __init__(self, start_token, end_token, pad_token, use_yt_categories, use_asr_subtitles, batch_size, device):
        self.start_token = start_token
        self.end_token = end_token
        self.pad_token = pad_token
        self.use_yt_categories = use_yt_categories
        self.use_asr_subtitles = use_asr_subtitles
        self.batch_size = batch_size
        self.device = device
    
    def createDataset(self, phase_meta_file_path, training_meta_file_path, min_occurance_freq=2):
        self._createFields(min_occurance_freq)
        
        tsv_fields = [
            ('video_id', None),
            ('caption', self.CAPTION_FIELD),
            ('start', None),
            ('end', None),
            ('duration', None)
        ]

        if self.use_yt_categories:
            tsv_fields.append(('category_32', self.CATEGORY_FIELD))
        else:
            tsv_fields.append(('category_32', None))
        if self.use_asr_subtitles:
            tsv_fields.append(('subs', self.ASR_SUBTITLES_FIELD))
        else:
            tsv_fields.append(('subs', None))
        tsv_fields.append(('phase', None))
        tsv_fields.append(('idx', self.INDEX_FIELD))

        self.phase_dataset = data.TabularDataset(
            path=phase_meta_file_path, format='tsv', skip_header=True, fields=tsv_fields,
            filter_pred=self.filter_callback)
        
        self.maximum_phase_caption_length = 0
        # Calculate maximum caption length
        for caption in self.phase_dataset.caption:
            if len(caption) > self.maximum_phase_caption_length:
                self.maximum_phase_caption_length = len(caption)
        
        if phase_meta_file_path != training_meta_file_path:
            self.training_dataset = data.TabularDataset(
                path=training_meta_file_path, format='tsv', skip_header=True, fields=tsv_fields,
                filter_pred=self.filter_callback)
        else:
            self.training_dataset = self.phase_dataset

        self.CAPTION_FIELD.build_vocab(self.training_dataset.caption, min_freq=min_occurance_freq)
        self.train_vocab = self.CAPTION_FIELD.vocab

        if self.use_asr_subtitles:
            self.ASR_SUBTITLES_FIELD.build_vocab(self.training_dataset.subs, min_freq=min_occurance_freq)
            self.train_subs_vocab = self.ASR_SUBTITLES_FIELD.vocab
        
        self.createDatasetIterator()

    def _createFields(self, min_occurance_freq):
        self.CAPTION_FIELD = data.ReversibleField(
            tokenize='spacy', init_token=self.start_token, 
            eos_token=self.end_token, pad_token=self.pad_token, lower=True, 
            batch_first=True, is_target=True, unk_token=UNKNOWN_TOKEN)

        self.INDEX_FIELD = data.Field(
            sequential=False, use_vocab=False, batch_first=True)

        if self.use_yt_categories:
            # preprocessing: if there is no category replace with -1 (unique number for dummy category)
            self.CATEGORY_FIELD = data.Field(
                sequential=False, use_vocab=False, batch_first=True, 
                preprocessing=data.Pipeline(lambda x: -1 if len(x) == 0 else int(float(x))))

            # filter the dataset if the a category is missing (31 -> 41 (count = 1 :()))
            self.filter_callback = lambda x: vars(x)['category_32'] != -1 and vars(x)['category_32'] != 31
        else:
            self.CATEGORY = None
            self.filter_callback = None

        if self.use_asr_subtitles:
            self.ASR_SUBTITLES_FIELD = data.ReversibleField(
                tokenize='spacy', init_token=self.start_token, 
                eos_token=self.end_token, pad_token=self.pad_token, lower=True, 
                batch_first=True, unk_token=UNKNOWN_TOKEN)
        else:
            self.ASR_SUBTITLES_FIELD = None
        
    def createDatasetIterator(self):
        sort_key = lambda x: 0
        self.dataset = data.BucketIterator(self.phase_dataset, self.batch_size, sort_key=sort_key, device=self.device, repeat=False, shuffle=True)
        self.dataset_iterator = iter(self.dataset)

    def getCaptionVocabSize(self):
        return len(self.train_vocab)
    
    def getTrainingCaptionVocabs(self):
        return self.train_vocab

    def getSubtitleVocabSize(self):
        return len(self.train_subs_vocab)
    
    def getTrainingSubtitleVocabs(self):
        return self.train_subs_vocab
    
    def getCaptionDataset(self):
        return self.dataset
    
    def getPaddingTokenIndex(self):
        self.pad_token_index = self.train_vocab.stoi[self.pad_token]
        return self.pad_token_index
    
    def getStartTokenIndex(self):
        self.start_token_index = self.train_vocab.stoi[self.start_token]
        return self.start_token_index
    
    def getEndTokenIndex(self):
        self.end_token_index = self.train_vocab.stoi[self.end_token]
        return self.end_token_index
    
    def getPhaseMaximumCaptionLength(self):
        return self.maximum_phase_caption_length
    
    def resetCaptionIterator(self):
        """
            This function should be called when epoch is ended and new epoch is starting
        """
        # This reset is required since we are using python iter function to iterate over captionDataset
        self.dataset_iterator = iter(self.dataset)

    def getNextBatchItems(self):
        """
            Returns single batch with sample size equal to batch_size
        """
        return next(self.dataset_iterator)

if __name__ == "__main__":
    TRAINING_META_FILE_PATH = 'C:/ACS/Master Thesis/Models/Video-Keyword-Extractor/multiModalDense/data/train_meta.csv'
    cd = CaptionDataset(START_TOKEN, END_TOKEN, PADDING_TOKEN, True, True, 28, 'cpu')
    cd.createDataset(TRAINING_META_FILE_PATH, TRAINING_META_FILE_PATH, 2)
    captionVocabSize = cd.getCaptionVocabSize()
    subtitlesVocabSize = cd.getSubtitleVocabSize()
    print('Caption vocab size: ' + str(captionVocabSize))
    print('Subtitles vocab size: ' + str(subtitlesVocabSize))

    dataset = cd.getCaptionDataset()
    print('Maximum phase caption length is: ' + str(cd.getPhaseMaximumCaptionLength()))
    print('Caption loader dataset batches (Each has 28 items): ' + str(len(dataset)))
    datasetIterator = iter(dataset)
    singleBatch = next(datasetIterator)
    print('Caption loader dataset single batch shape: ' + str(singleBatch.batch_size))
    captionData = singleBatch.caption
    captionData_np = np.array(captionData)
    print(captionData_np.shape)
    print(captionData_np[0])
    batchVideoIndex = singleBatch.idx
    print(batchVideoIndex)
    for index in batchVideoIndex:
        idx = index.item()
        # print(idx)
    pass
