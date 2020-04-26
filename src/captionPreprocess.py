import string

START_KEYWORD = '>'
STOP_KEYWORD = '<'
NONE_KEYWORD = '?!?'
EXTRA_KEYWORD = '___' # extra keyword for unseen vocab words for validation and test
punctuation_mapping = str.maketrans('', '', string.punctuation)

class CaptionPreprocessor:
    def __init__(self, caption_inputs :dict, captionCleanup = True, word_freq_threshold=3):
        self.caption_inputs = caption_inputs
        self.word_freq_threshold = word_freq_threshold
        self.captionCleanup = captionCleanup
        if self.captionCleanup is True:
            self.cleanupCaptions()
        #self.appendStartStop()
        self.extractWordList()
        self.filterMostOccuringWords()
        self.createWordIndexDictionary()

    def cleanupCaptions(self):
        cleaned_caption_inputs = dict()
        self.caption_max_length = 0
        for id, captions in self.caption_inputs.items():
            if id not in cleaned_caption_inputs:
                cleaned_caption_inputs[id] = list()
            for caption in captions:
                caption_words = caption.split()
                caption_words = [word.lower() for word in caption_words]
                caption_words = [word.translate(punctuation_mapping) for word in caption_words]
                caption_words = [word for word in caption_words if len(word)>1]
                caption_words = [word for word in caption_words if word.isalpha()]
                cleaned_caption_inputs[id].append(' '.join(caption_words))
                self.caption_max_length = max(self.caption_max_length, len(caption_words))
        self.caption_inputs = cleaned_caption_inputs


    def appendStartStop(self):
        # Append START_KEYWORD and END_KEYWORD
        for id, captions in self.caption_inputs.items():
            # For each captions for a video, append start and stop keyword
            for index in range(len(captions)):
                captions[index] = f'{START_KEYWORD} {captions[index]} {STOP_KEYWORD}'

        # Update max-length of caption by 2 for start and stop index
        self.caption_max_length += 2 
    
    def extractWordList(self):
        self.all_captions = []
        for id, captions in self.caption_inputs.items():
            for caption in captions:
                self.all_captions.append(caption)

        self.words_occurance_count = dict()
        self.unique_words = set()
        for caption in self.all_captions:
            for word in caption.split(' '):
                self.unique_words.add(word)
                self.words_occurance_count[word] = self.words_occurance_count.get(word, 0) + 1

    def filterMostOccuringWords(self):
        actual_word_vocab = [w for w in self.words_occurance_count if self.words_occurance_count[w] >= self.word_freq_threshold]
        self.final_captions_vocab = list()
        self.final_captions_vocab.append(START_KEYWORD)
        self.final_captions_vocab.append(STOP_KEYWORD)
        self.final_captions_vocab.append(NONE_KEYWORD)
        self.final_captions_vocab.append(EXTRA_KEYWORD)
        for word in actual_word_vocab:
            if word in [NONE_KEYWORD, START_KEYWORD, STOP_KEYWORD, EXTRA_KEYWORD]:
                continue
            self.final_captions_vocab.append(word)

    def createWordIndexDictionary(self):
        self.index_to_word = dict()
        self.word_to_index = dict()

        index = 0
        for word in self.final_captions_vocab:
            self.index_to_word[index] = word
            self.word_to_index[word] = index
            index += 1

        # adding 1 to vocab size as we have 0 index reserved for dummy values
        self.final_vocab_size = len(self.index_to_word)

    def getVocabSize(self):
        return self.final_vocab_size
    
    def getWordToIndexDict(self):
        return self.word_to_index

    def getIndexToWordDict(self):
        return self.index_to_word
    
    def getCaptionsVocabList(self):
        return self.final_captions_vocab
    
    def getFilteredWord(self, word):
        if word in self.word_to_index.keys():
            return word
        return EXTRA_KEYWORD
    




