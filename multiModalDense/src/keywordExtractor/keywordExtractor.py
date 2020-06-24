import keywordExtractor.keywordExtractorUtil as util
import keywordExtractor.algorithmBuilder as builder

class KeywordExtractor:

    def __init__(self, algorithm='YAKE'):
        self.algorithm = algorithm

    def getKeywords(self, top_n=10, video_description=None, video_ids=None, video_description_dataset_file_path=None):
        if self.keywordExtractor is None:
            self.keywordExtractor = builder.createInstanceOfPkeAlgo()
        
        if video_description_dataset_file_path is None and video_ids is None and video_description is None:
            raise Exception('Please provide valid input document in form of files')
            
        if video_ids is not None and video_description_dataset_file_path is None:
            raise Exception('Please provide video description file path as input')
        
        if video_description_dataset_file_path is not None:
            self.extractKeywordForDataset(video_description_dataset_file_path, video_ids)
        else:
            self.extractKeywordFromVideoDescription(video_description)
    

    def extractKeywordFromVideoDescription(self, video_description_input):
        # load text content
        self.keywordExtractor.load_document(input=video_description_input, language='en')

        self.keywordExtractor.candidate_selection()

        self.keywordExtractor.candidate_weighting()

        # select N-Best keyphrases
        keyphrases = self.keywordExtractor.get_n_best(n=10)

        return keyphrases
    
    def extractKeywordForDataset(self, video_description_dataset_file_path, video_ids):
        pass

    
    