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

def filterCaptionsForSamples(caption_file, video_ids):
    all_english_captions = loadCaptionData(caption_file)
    filteredCaptions = dict()
    for videoId in video_ids:
        filteredCaptions[videoId] = all_english_captions[videoId]
    return filteredCaptions

if __name__ == '__main__':
    allCaptions = loadCaptionData(config.CSV_FILE_PATH)
    print(len(allCaptions))
    print(len(allCaptions['mv89psg6zh4_33_46']))
