import keywordExtractor.keywordExtractorUtil as util
import keywordExtractor.algorithmBuilder as builder
import json
import re
from tqdm import tqdm

class KeywordExtractor:

    def __init__(self, algorithm='YAKE'):
        self.algorithm = algorithm

    def getKeywords(self, top_n=10, video_description=None, video_ids=None, video_description_dataset_file_path=None, output_file_path=None):
        
        if video_description_dataset_file_path is None and video_ids is None and video_description is None:
            raise Exception('Please provide valid input document in form of files')
            
        if video_ids is not None and video_description_dataset_file_path is None:
            raise Exception('Please provide video description file path as input')
        
        if video_ids is None and video_description_dataset_file_path is not None and output_file_path is None:
            raise Exception('Please provide output file path to store extracted keywords')

        if video_description_dataset_file_path is not None:
            return self.extractKeywordForDataset(video_description_dataset_file_path, video_ids, output_file_path)

        return self.extractKeywordFromVideoDescription(video_description)
    

    def extractKeywordFromVideoDescription(self, video_description_input):
        keyphrases = self.getKeyphrasesFromText(video_description_input)

        return dict({
            'no_video_id': {
                'multiModal' : keyphrases
            }
        })
    
    def extractKeywordForDataset(self, video_description_dataset_file_path, video_ids, output_file_path):
        result_dict = dict()
        dataset_dict = util.readPredictionJSON(video_description_dataset_file_path)
        writeToFile = True
        if video_ids:
            writeToFile = False
        for video_id in tqdm(list(dataset_dict.keys())):
            if video_ids and video_id not in video_ids:
                continue
            
            modifiedCaptionKeyphrases, captionTextInput = self.getVideoWiseKeyphrases(video_id, dataset_dict[video_id]['sentences'])
            modifiedSubtitlesKeyphrases, subtitleTextInput = self.getVideoWiseKeyphrases(video_id, dataset_dict[video_id]['subtitles'])
            if len(captionTextInput) == 0  and len(subtitleTextInput) == 0:
                print(f'No keyphrases for {video_id}')
                continue
            try:
                multiModalKeyphrases = self.getKeyphrasesFromText(f'{captionTextInput}. {subtitleTextInput}')
            except:
                print(f'Error occurred while extracting keywords for {video_id}')
                print(f'Concatenated multimodal text: {captionTextInput}. {subtitleTextInput}')
                raise
            
            modifiedMultiModalKeyphrases = []
            for keyphrase in multiModalKeyphrases:
                modifiedMultiModalKeyphrases.append(list(keyphrase))

            result_dict[video_id] = {
                'originalCaption': captionTextInput,
                'originalSubtitles': subtitleTextInput,
                'captions': modifiedCaptionKeyphrases,
                'subtitles': modifiedSubtitlesKeyphrases,
                'multiModal': modifiedMultiModalKeyphrases
            }
        if writeToFile is False:
            return result_dict

        with open(output_file_path, 'w') as outf:
            json.dump(result_dict, outf)
            print(f'Keyword extracted and stored at {output_file_path}')
    
    def getVideoWiseKeyphrases(self, videoId, listOfTexts):
        filtered_texts = []
        for text in listOfTexts:
            text = text.strip()
            if len(text) > 0:
                filtered_texts.append(text)
        if len(filtered_texts) == 0:
            return [], ''
        concatenatedInput = '. '.join(filtered_texts)
        concatenatedInput = re.sub(r'[",]', '', concatenatedInput)
        try:
            textKeyphrases = self.getKeyphrasesFromText(concatenatedInput)
        except:
            print(f'Error occurred while extracting keywords for {videoId}')
            print(f'Concatenated text: {concatenatedInput}')
            raise
        
        modifiedTextKeyphrases = []
        for keyphrase in textKeyphrases:
            modifiedTextKeyphrases.append(list(keyphrase))
        return modifiedTextKeyphrases, concatenatedInput

    def getKeyphrasesFromText(self, text):
        self.keywordExtractor = builder.createInstanceOfPkeAlgo(self.algorithm)

        # load text content
        self.keywordExtractor.load_document(input=text, language='en')

        self.keywordExtractor.candidate_selection()

        self.keywordExtractor.candidate_weighting()

        # select N-Best keyphrases
        keyphrases = self.keywordExtractor.get_n_best(n=10)

        return keyphrases

    
    