import json

import config
import keywordExtractor.keywordExtractorUtil as util
from keywordExtractor.keywordExtractor import KeywordExtractor

caption_dict, _ = util.readPredictedJSON(config.PREDICTED_VAL_1_JSON, config.VALIDATION_1_META_FILE_PATH)
metadata_dict = util.readMetaCSVtoDict(config.VALIDATION_1_META_FILE_PATH)
missing_video_ids = util.prepareFinalDataset(caption_dict, metadata_dict)

with open(config.VIDEO_DESCRIPTION_VALIDATION_1_JSON, 'w') as outf:
    json.dump(caption_dict, outf)

ke = KeywordExtractor(config.KEYWORD_EXTRACTOR_ALGORITHM)
ke.getKeywords(top_n=config.N_BEST_KEYWORDS, video_description_dataset_file_path=config.VIDEO_DESCRIPTION_VALIDATION_1_JSON,\
     output_file_path=config.KEYWORD_EXTRACTOR_OUTPUT_JSON)

# result_dict = ke.getKeywords(top_n=config.N_BEST_KEYWORDS, video_description='A person is seen riding along a water skis while the camera captures him from several angles. The boat is pulling the rope as they go through the water. The people continue riding around on the boat while the camera captures them from several angles')

# result_dict = ke.getKeywords(top_n=config.N_BEST_KEYWORDS, video_ids=config.KEYWORD_EXTRACTION_SAMPLE_VIDEO_IDS, \
#     video_description_dataset_file_path=config.VIDEO_DESCRIPTION_VALIDATION_1_JSON)

# print(result_dict)
