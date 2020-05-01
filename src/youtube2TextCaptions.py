import csv
import config
import random

def loadCaptionData(captionFileName):
    englishCaptions = dict()
    with open(captionFileName, newline='', encoding="utf8") as csvFile:
        csvDataReader = csv.DictReader(csvFile)
        for row in csvDataReader:
            if row['Language'] != 'English':
                continue
            row_id = row['VideoID'] + '_' + row['Start'] + '_' + row['End']
            if row_id not in englishCaptions:
                englishCaptions[row_id] = list()
            caption = row['Description']
            englishCaptions[row_id].append(caption)
    return englishCaptions

def filterCaptionsForSamples(caption_file, video_ids=None, load_single_caption=True, caption_per_video_limit=4):
    SAFETY_THRESHOLD = 3
    all_english_captions = loadCaptionData(caption_file)
    filteredCaptions = dict()
    if video_ids is None:
        video_ids = list(all_english_captions.keys())
    for videoId in video_ids:
        if load_single_caption is False:
            if(len(all_english_captions[videoId]) > caption_per_video_limit):
                all_video_captions = all_english_captions[videoId]
                all_video_captions.sort(key = lambda s: len(s.split(' ')))
                if (len(all_english_captions[videoId]) >= (caption_per_video_limit + SAFETY_THRESHOLD)):
                    # Get 4th highest to 8th
                    start_index = len(all_english_captions[videoId]) - (caption_per_video_limit + SAFETY_THRESHOLD)
                    end_index = len(all_english_captions[videoId]) - SAFETY_THRESHOLD
                    filteredCaptions[videoId] = all_video_captions[start_index:end_index]
                else:
                    # Get last 4 captions
                    filteredCaptions[videoId] = all_video_captions[-caption_per_video_limit:]
            else:
                filteredCaptions[videoId] = all_english_captions[videoId]
            random.shuffle(filteredCaptions[videoId])
            continue
        max_length_caption = None
        max_length = 0
        for caption in all_english_captions[videoId]:
            if len(caption.split(' ')) > max_length:
                max_length_caption = caption
                max_length = len(caption.split(' '))
        filteredCaptions[videoId] = [max_length_caption]
        random.shuffle(filteredCaptions[videoId])
    return filteredCaptions

if __name__ == '__main__':
    #allCaptions = loadCaptionData(config.CSV_FILE_PATH)
    #allCaptions = filterCaptionsForSamples(config.CSV_FILE_PATH, ['ibSwITK4jjQ_14_24'], load_single_caption=False)
    allCaptions = filterCaptionsForSamples(config.CSV_FILE_PATH, load_single_caption=False)    
    print(len(allCaptions))
    print(len(allCaptions['ibSwITK4jjQ_14_24']))
    print(len(allCaptions['ibSwITK4jjQ_14_24'][0]))

