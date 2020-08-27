# Video-Keyword-Extractor
A Master Thesis Project on Video Keyword Extractor using Video Summarization techniques.

# Pipeline
The pipeline consists of 3 stages,
1. Feature extraction from video
2. Video captioning using all modalities
3. Keyword extraction from caption and subtitles

![Pipeline Architecture](https://github.com/VP-0822/Video-Keyword-Extractor/blob/master/images/pipelineDetailed.png)

## Video Captioning
Author has implemented following 2 techniques for video captioning
### RNN-based Video Captioning Model
The architecture of RNN-based video captioning model is as below. The model was trained using the YouTube2Text dataset.
![RNN based Architecture](https://github.com/VP-0822/Video-Keyword-Extractor/blob/master/images/basicRNNModel.png)

### Multi-modal Deep Video Captioning Model
The author have initially implemented the model referring the <a href="https://arxiv.org/abs/2003.07758">Iashin et al.</a> paper. The following image shows the architecture of the MDVC model. The model was trained using the ActivityNet dataset.
![MDVC Architecture](https://github.com/VP-0822/Video-Keyword-Extractor/blob/master/images/MDVC.png)

Based on above approach the author of the thesis proposed new model to improve the transformer by encoding visual and audio modality inputs together, using the technique proposed by the <a href="https://arxiv.org/abs/1809.04938">LiveBot</a>. The intention was to improve the caption sentence quality. For e.g. if in video 2 people are having conversations related to animals, The current state-of-the-art has the caption 'Two people are talking in the room.' , What the caption should be 'Two people are talking about animals in the room.' Below image shows the modified audio encoder,
![MDVC Variant Architecture](https://github.com/VP-0822/Video-Keyword-Extractor/blob/master/images/MDVCaudioEncoder.png)

## Video Keyword Extraction
Author has added a program to extract keywords from the generated captions from the above model and subtitles using the YouTube ASR technique. Author has used python toolkit for keyword extraction, <a href="https://github.com/boudinfl/pke">pke</a>.

## Results
Video Id: kXbc9D0sF5k (https://www.youtube.com/watch?v=kXbc9D0sF5k)
### Ground Truth Captions
**0 Sec – 37 Sec:** *People are seen shoveling snow in several clips as well as getting a camera ready.*  
**31 Sec – 130 Sec:** *Many people speak to the camera as people ski around public places.*  
**103 Sec – 190 Sec:** *People perform jumps and tricks while sometimes falling and continuing to speak to the camera.*   

### Using <a href="https://arxiv.org/abs/2003.07758">Iashin et al.</a>
**0 Sec – 37 Sec:** *A man is seen speaking to the camera and leads into several clips of people riding down the hill.*  
**31 Sec – 130 Sec:** *A man is seen speaking to the camera and leads into clips of him riding down a hill.*  
**103 Sec – 190 Sec:** *The man then jumps over a hill and jumps over a hill.*  

### Using proposed model
**0 Sec – 37 Sec:** *The man is seen walking around the snow and speaking to the camera.*  
**31 Sec – 130 Sec:** *The man then continues to speak to the camera while more shots of the camera and ends with several people riding down the hill.*  
**103 Sec – 190 Sec:** *The man then is snowboarding and ends by speaking to the camera.*  

### Keywords using TextRank

*Using captions only:*   

**0 Sec – 37 Sec:** *'camera', 'man', 'snow'*  
**31 Sec – 130 Sec:** *'several people', 'more shots', 'hill', 'camera'*  
**103 Sec – 190 Sec:** *'camera', 'man'*  
**For the entire video:** *'several people', 'hill', 'man', 'camera', 'snow' *  

*Using Subtitle and Captions:*    

**0 Sec – 37 Sec:** *'urban skiing', 'city limits', 'fun', 'snow', 'backcountry'*  
**31 Sec – 130 Sec:** *'professional urban ski', 'urban community', 'young guys', 'high schools'*   
**103 Sec – 190 Sec:** *'professional urban ski', 'mountain sport', 'different spin', 'young guys', 'skiing' *  
**For the entire video:** *'professional urban ski', 'urban skiing', 'urban community', 'mountain sport', 'young guys'*  

## References
1. Iashin et al. https://arxiv.org/abs/2003.07758
2. LiveBot https://arxiv.org/abs/1809.04938
3. YouTube2Text Dataset. Chen, David & Dolan, William. (2011). Collecting Highly Parallel Data for Paraphrase Evaluation. 190-200.
4. ActivityNet Dataset. Online. [Cited on 10.06.2020] http://activity-net.org/download.html
5. https://github.com/boudinfl/pke
6. https://github.com/scopeInfinity/Video2Description
