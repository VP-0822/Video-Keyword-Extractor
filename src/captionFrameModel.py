import numpy as np
import captionPreprocess as cp
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.sequence import pad_sequences

class CaptionFrameModel:
    def __init__(self, final_caption_length, total_vocab_size):
        self.final_caption_length = final_caption_length
        self.total_vocab_size = total_vocab_size
    
    def _dataGenerator(self, training_set, wordtoidx, num_samples_per_batch, dataset_name='Training'):
        # x1 - Training data for videos
        # x2 - The caption that goes with each photo
        # y - The predicted rest of the caption
        x1, x2, y = [], [], []
        current_batch_item_count=0
        #print('Dataset name: ' + dataset_name)
        while True:
            for key, values in training_set.items():
                current_batch_item_count+=1
                frame_features = values[0]
                single_video_captions = values[1]
                in_caption_item = f'{cp.START_KEYWORD} {single_video_captions}'
                in_seq = [wordtoidx[word] for word in in_caption_item.split(' ') if word in wordtoidx]
                for i in range(self.final_caption_length-len(in_seq)):
                    in_seq.append(wordtoidx[cp.NONE_KEYWORD])

                out_caption_item = f'{single_video_captions} {cp.STOP_KEYWORD}'
                out_caption_seq = [wordtoidx[word] for word in out_caption_item.split(' ') if word in wordtoidx]
                for i in range(self.final_caption_length-len(out_caption_seq)):
                    out_caption_seq.append(wordtoidx[cp.NONE_KEYWORD])
                out_seq = list()
                for count in range(self.final_caption_length):
                    out_seq.append(to_categorical([out_caption_seq[count]], num_classes=self.total_vocab_size)[0])
                x1.append(in_seq)
                x2.append(frame_features)
                y.append(out_seq)
                if current_batch_item_count==num_samples_per_batch:
                    # print('##########################')
                    # print('Dataset name: ' + dataset_name)
                    # print("=== x1 ***********************************====") 
                    # print(', '.join(map(str,np.array(x1).shape)) +' && '+ ', '.join(map(str,np.array(x2).shape)) +' && '+ ', '.join(map(str,np.array(y).shape)))
                    # print(np.array(x1[0]).shape)
                    # print("=== x2 ***********************************====")
                    # print(np.array(x2).shape)
                    # print(np.array(x2[0]).shape)
                    yield ([np.array(x1), np.array(x2)], np.array(y))
                    current_batch_item_count=1
                    x1, x2, y = [], [], []
    
    def _predictFromModel(self, model, video_sample, wordtoidx, idxtoword):
        video_frame_input = video_sample[0]
        original_video_caption_input = video_sample[1][0]
        dummy_caption = [cp.START_KEYWORD]
        dummy_caption.extend([cp.STOP_KEYWORD] * (self.final_caption_length - 1))
        #dummy_caption = f'{cp.START_KEYWORD} {original_video_caption_input}'
        video_dummy_caption = [wordtoidx[word] for word in dummy_caption if word in wordtoidx]
        # for i in range(max_caption_length-len(video_dummy_caption)):
        #     video_dummy_caption.append(wordtoidx[cp.NONE_KEYWORD])
        #print(video_dummy_caption)
        input_sequence = pad_sequences([video_dummy_caption], maxlen=self.final_caption_length)
        #print(list(input_sequence))
        #print(np.argmax(captionoutput[0][3]))
        #print(np.argmax(captionoutput[0][7]))
        input_seq_list = list(input_sequence)
        output_caption = []
        caption_length_counter = 0
        while caption_length_counter < self.final_caption_length:
            captionoutput = model.predict([np.array(input_seq_list), np.array([video_frame_input])])
            #print('Shape of predict model: ' + str(captionoutput.shape))
            yhat = np.argmax(captionoutput[0][caption_length_counter])
            word = idxtoword[yhat]
            if word == cp.STOP_KEYWORD:
                break
            output_caption.append(word)
            if caption_length_counter + 1 != self.final_caption_length:
                input_seq_list[0][caption_length_counter+1] = wordtoidx[word]
            caption_length_counter += 1
            #for i, newOneHotWord in enumerate()
        output_caption_text = ' '.join(output_caption)
        return original_video_caption_input, output_caption_text
    
