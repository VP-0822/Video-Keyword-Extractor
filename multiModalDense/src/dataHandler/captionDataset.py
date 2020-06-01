import numpy as np
from torchtext import data
'''
    Uses torchtext to create TabularDataset. (Docs: https://pytorch.org/text/index.html)
'''
class CaptionDataset:
    def __init__(self, start_token, end_token, pad_token, use_yt_categories, use_asr_subtitles):
        self.start_token = start_token
        self.end_token = end_token
        self.pad_token = pad_token
        self.use_yt_categories = use_yt_categories
        self.use_asr_subtitles = use_asr_subtitles
    
    def createDataset(self, phase_meta_file_path, training_meta_file_path):
        self._createFields()
        
        tsv_fields = [
            ('video_id', None),
            ('caption', self.CAPTION_FIELD),
            ('start', None),
            ('end', None),
            ('duration', None),
            ('category_32', self.CATEGORY_FIELD),
            ('subs', self.ASR_SUBTITLES_FIELD),
            ('phase', None),
            ('idx', self.INDEX_FIELD),
        ]

        self.phase_dataset = data.TabularDataset(
            path=phase_meta_file_path, format='tsv', skip_header=True, fields=tsv_fields,
            filter_pred=self.filter_callback)
        
        if phase_meta_file_path != training_meta_file_path:
            self.training_dataset = data.TabularDataset(
                path=training_meta_file_path, format='tsv', skip_header=True, fields=tsv_fields,
                filter_pred=self.filter_callback)
        else:
            self.training_dataset = self.phase_dataset
        
    def getCaptionVocabSize(self, min_occurance_freq):
        self.CAPTION_FIELD.build_vocab(self.phase_dataset.caption, min_freq=min_occurance_freq)
        self.train_vocab = self.CAPTION_FIELD.vocab
        return len(self.train_vocab)
    
    def getSubtitleVocabSize(self, min_occurance_freq):
        self.ASR_SUBTITLES_FIELD.build_vocab(self.phase_dataset.subs, min_freq=min_occurance_freq)
        self.train_subs_vocab = self.ASR_SUBTITLES_FIELD.vocab
        return len(self.train_subs_vocab)
    
    def getCaptionDatasetIterator(self, batch_size, tensor_device):
        sort_key = lambda x: 0
        datasetIterator = data.BucketIterator(
            self.phase_dataset, batch_size, sort_key=sort_key, device=tensor_device, repeat=False, shuffle=True)
        return datasetIterator
    
    def getPaddingTokenIndex(self):
        self.pad_token_index = self.train_vocab.stoi[self.pad_token]
        return self.pad_token_index
    
    def getStartTokenIndex(self):
        self.start_token_index = self.train_vocab.stoi[self.start_token]
        return self.start_token_index
    
    def getEndTokenIndex(self):
        self.end_token_index = self.train_vocab.stoi[self.end_token]
        return self.end_token_index

    def _createFields(self):
        self.CAPTION_FIELD = data.ReversibleField(
            tokenize='spacy', init_token=self.start_token, 
            eos_token=self.end_token, pad_token=self.pad_token, lower=True, 
            batch_first=True, is_target=True)

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
                batch_first=True)
        else:
            self.ASR_SUBTITLES_FIELD = None

if __name__ == "__main__":
    pass
