import csv
import config

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

def filterCaptionsForSamples(caption_file, video_ids, load_single_caption=True, caption_per_video_limit=4):
    all_english_captions = loadCaptionData(caption_file)
    filteredCaptions = dict()
    for videoId in video_ids:
        if load_single_caption is False:
            if(len(all_english_captions[videoId]) > caption_per_video_limit):
                filteredCaptions[videoId] = all_english_captions[videoId][:caption_per_video_limit]
            else:
                filteredCaptions[videoId] = all_english_captions[videoId]
            continue
        max_length_caption = None
        max_length = 0
        for caption in all_english_captions[videoId]:
            if len(caption.split(' ')) > max_length:
                max_length_caption = caption
                max_length = len(caption.split(' '))
        filteredCaptions[videoId] = [max_length_caption]
    return filteredCaptions

if __name__ == '__main__':
    #allCaptions = loadCaptionData(config.CSV_FILE_PATH)
    allCaptions = filterCaptionsForSamples(config.CSV_FILE_PATH, ['mv89psg6zh4_33_46'])
    print(len(allCaptions))
    print(len(allCaptions['mv89psg6zh4_33_46']))
    print(len(allCaptions['mv89psg6zh4_33_46'][0]))

