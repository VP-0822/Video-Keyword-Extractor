import csv
import random
import os
# from captionPreprocess import CaptionPreprocessor

class Youtube2TextCaptions:
    def __init__(self, captionFileName, video_ids=None, caption_per_video_limit=4):
        self.captionFileName = captionFileName
        self.sample_video_ids = video_ids
        self.caption_per_video_limit = caption_per_video_limit
        self.SAFETY_THRESHOLD = 3
        self._loadCaptionData()
        self.filterCaptionsForSamples()
    
    def _loadCaptionData(self):
        self.englishCaptions = dict()
        with open(self.captionFileName, newline='', encoding="utf8") as csvFile:
            csvDataReader = csv.DictReader(csvFile)
            for row in csvDataReader:
                if row['Language'] != 'English':
                    continue
                row_id = row['VideoID'] + '_' + row['Start'] + '_' + row['End']
                if row_id not in self.englishCaptions:
                    self.englishCaptions[row_id] = list()
                caption = row['Description']
                self.englishCaptions[row_id].append(caption)

    def filterCaptionsForSamples(self):
        self.filteredCaptions = dict()
        if self.sample_video_ids is None:
            self.sample_video_ids = list(self.englishCaptions.keys())
        for videoId in self.sample_video_ids:
            if self.caption_per_video_limit > 1:
                if(len(self.englishCaptions[videoId]) > self.caption_per_video_limit):
                    all_video_captions = self.englishCaptions[videoId]
                    all_video_captions.sort(key = lambda s: len(s.split(' ')))
                    if (len(self.englishCaptions[videoId]) >= (self.caption_per_video_limit + self.SAFETY_THRESHOLD)):
                        # Get 4th highest to 8th
                        start_index = len(self.englishCaptions[videoId]) - (self.caption_per_video_limit + self.SAFETY_THRESHOLD)
                        end_index = len(self.englishCaptions[videoId]) - self.SAFETY_THRESHOLD
                        self.filteredCaptions[videoId] = all_video_captions[start_index:end_index]
                    else:
                        # Get last 4 captions
                        self.filteredCaptions[videoId] = all_video_captions[-self.caption_per_video_limit:]
                else:
                    self.filteredCaptions[videoId] = self.englishCaptions[videoId]
                #random.shuffle(filteredCaptions[videoId])
                continue
            all_video_captions = self.englishCaptions[videoId]
            all_video_captions.sort(key = lambda s: len(s.split(' ')))
            video_captions_len = len(all_video_captions)
            self.filteredCaptions[videoId] = [all_video_captions[video_captions_len-3]]
            #random.shuffle(filteredCaptions[videoId])
    
    def getFilteredCaptions(self):
        return self.filteredCaptions

if __name__ == '__main__':
    root_dir = os.path.dirname(__file__)
    CSV_FILE_PATH = os.path.join(root_dir,'../data/youtube2TextCaptions/MSR_Video_Description_Corpus.csv')
    y2t = Youtube2TextCaptions(CSV_FILE_PATH, caption_per_video_limit=1)
    allCaptions = y2t.getFilteredCaptions()  
    print(len(allCaptions))
    print(len(allCaptions['ibSwITK4jjQ_14_24']))
    print(len(allCaptions['ibSwITK4jjQ_14_24'][0]))
    print([len(a[0]) for k,a in allCaptions.items()])
    #allCaptions = loadCaptionData(config.CSV_FILE_PATH)
    #allCaptions = filterCaptionsForSamples(config.CSV_FILE_PATH, ['ibSwITK4jjQ_14_24'], load_single_caption=True)
    #cp = CaptionPreprocessor(allCaptions, word_freq_threshold=1)
    #print('Total Vocab size: ' + str(cp.getVocabSize()))
