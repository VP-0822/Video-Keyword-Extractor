import json
from torchtext import data

def readPredictionJSON(predicted_json_file_path):
    with open(predicted_json_file_path) as json_file:
        data_dict = json.load(json_file)
        return data_dict

def readMetaCSVtoDict(meta_csv_file_path):
    tsv_fields = [
            ('video_id', data.Field()),
            ('caption', None),
            ('start', data.Field()),
            ('end', data.Field()),
            ('duration', data.Field()),
            ('category_32', None),
            ('subs', data.Field()),
            ('phase', None),
            ('idx', data.Field())]
    meta_data_dataset = data.TabularDataset(
                path=meta_csv_file_path, format='tsv', skip_header=True, fields=tsv_fields)
    data_examples = meta_data_dataset.examples
    return_dict = {}
    for example in data_examples:
        if example.video_id[0] in list(return_dict.keys()):
            duration = float(example.duration[0])
            start = float(example.start[0])
            end = float(example.end[0])
            timestamps = [start, end]
            return_dict[example.video_id[0]]['timestamps'].append(timestamps)
            return_dict[example.video_id[0]]['subtitles'].append(' '.join(example.subs))
        else:
            duration = float(example.duration[0])
            start = float(example.start[0])
            end = float(example.end[0])
            timestamps = [start, end]
            timestamps_list = [timestamps]
            subtitle = ' '.join(example.subs)
            subtitles_list = [subtitle]
            return_dict[example.video_id[0]] = {
                "duration" : duration,
                "timestamps": timestamps_list,
                "subtitles": subtitles_list
            }
    return return_dict

def readPredictedJSON(predicted_caption_file_path, meta_csv_file_path):
    missing_video_ids = []
    predicted_caption_dict = readPredictionJSON(predicted_caption_file_path)
    predicted_results = predicted_caption_dict['results']
    metadata_dict = readMetaCSVtoDict(meta_csv_file_path)
    return_dict = {}
    for video_id in list(predicted_results.keys()):
        if video_id not in list(metadata_dict.keys()):
            missing_video_ids.append(video_id)
            continue
        video_duration = metadata_dict[video_id]['duration']
        video_results = predicted_results[video_id]
        timestamps = []
        captions = []
        for video_segment_result in video_results:
            predicted_caption = video_segment_result['sentence']
            timestamp = video_segment_result['timestamp']
            timestamps.append(timestamp)
            captions.append(predicted_caption)
        metadata_timestamps = metadata_dict[video_id]['timestamps']
        if len(metadata_timestamps) != len(timestamps):
            raise Exception('Found mismatch in timestamps between predicted result and metadata')
        modified_timestamps = []
        modified_captions = []
        for index, original_timestamp in enumerate(metadata_timestamps):
            original_timestamp_found = False
            for predicted_index, predicted_timestamp in enumerate(timestamps):
                if original_timestamp[0] == predicted_timestamp[0] and original_timestamp[1] == predicted_timestamp[1]:
                    modified_timestamps.append(predicted_timestamp)
                    modified_captions.append(captions[predicted_index])
                    original_timestamp_found = True
            if original_timestamp_found is False:
                print(video_id)
                raise Exception('Mismatched start/end time for metadata and predicted result timestamps')

        return_dict[video_id] = {
            "duration" : video_duration,
            "timestamps": modified_timestamps,
            "sentences": modified_captions
        }
    return return_dict, missing_video_ids

def prepareFinalDataset(captionDict, subtitlesDict):
    missing_video_ids = []
    for video_id in list(captionDict.keys()):
        if video_id not in list(subtitlesDict.keys()):
            missing_video_ids.append(video_id)
            continue
        if len(subtitlesDict[video_id]['timestamps']) != len(captionDict[video_id]['timestamps']):
            print(f'mismatched timestamps for video id: {video_id}')
            raise Exception('Invalid data')
        captionDict[video_id]['subtitles'] = subtitlesDict[video_id]['subtitles']
    return missing_video_ids
    
if __name__ == "__main__":
    VAL_1_META_FILE_PATH = 'C:/ACS/MasterThesis/Models/Video-Keyword-Extractor/multiModalDense/data/val_1_meta.csv'
    VAL_1_PREDICTION_JSON_FILE_PATH = 'C:/ACS/MasterThesis/Models/Video-Keyword-Extractor/multiModalDense/data/predictions_validate_1_epoch31.json'
    # VAL_1_RESULTS_JSON_FILE_PATH = 'C:/ACS/MasterThesis/Models/Video-Keyword-Extractor/multiModalDense/data/val_1_no_missings.json'
    subtitle_dict = readMetaCSVtoDict(VAL_1_META_FILE_PATH)
    # print()
    # caption_dict = readPredictionJSON(VAL_1_RESULTS_JSON_FILE_PATH)

    return_dict, missing_video_ids = readPredictedJSON(VAL_1_PREDICTION_JSON_FILE_PATH, VAL_1_META_FILE_PATH)
    missing_video_ids = prepareFinalDataset(return_dict, subtitle_dict)
    with open('test_file.json', 'w') as outf:
        json.dump(return_dict, outf)
    if len(missing_video_ids) == 0:
        for video_key in list(return_dict.keys()):
            print(return_dict[video_key])
            break
    else:
        print(missing_video_ids)
        print(len(missing_video_ids))
    
    