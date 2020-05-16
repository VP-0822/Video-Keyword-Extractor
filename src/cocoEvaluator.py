from pycocoevalcap.tokenizer.ptbtokenizer import PTBTokenizer
from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.meteor.meteor import Meteor
from pycocoevalcap.rouge.rouge import Rouge
from pycocoevalcap.cider.cider import Cider

class COCOCaptionEvaluator:
    def __init__(self, dataset):
        self.dataset = dataset
        self.evalResults = dict()
        self.evalVideoResult = dict()
        self.loadDataset()
        self.initializeTokenizer()

    def loadDataset(self):
        self.videoIds = list()
        self.actualCaptions = dict()
        self.predictedCaptions = dict()
        for videoId, captions in self.dataset.items():
            self.videoIds.append(videoId)
            self.actualCaptions[videoId] = [{'caption': captions[0]}]
            self.predictedCaptions[videoId] = [{'caption': captions[1]}]
    
    def initializeTokenizer(self):
        groundTruthCaptions = self.actualCaptions
        predictedCaptions = self.predictedCaptions

        #Tokenize
        tokenizer = PTBTokenizer()
        self.gtc_tokens = tokenizer.tokenize(groundTruthCaptions)
        self.pc_tokens = tokenizer.tokenize(predictedCaptions)

    def computeBleuScore(self):
        methods = ["Blue 1.0", "Blue 2.0", "Blue 3.0", "Blue 4.0"]

        #Compute Score
        scores, blueList  = Bleu(4).compute_score(self.gtc_tokens, self.pc_tokens)

        for score, blue, method in zip(scores, blueList, methods):
            self.evalResults[method] = score
            self.setVideoEvalResults(blue, method)

    def computeMeteorScore(self):
        score, meteorVideoScores = Meteor().compute_score(self.gtc_tokens, self.pc_tokens)

        self.evalResults['Meteor'] = score
        self.setVideoEvalResults(meteorVideoScores, 'Meteor')
    
    def computeRoughScore(self):
        score, roughVideoScores = Rouge().compute_score(self.gtc_tokens, self.pc_tokens)

        self.evalResults['Rough_L'] = score
        self.setVideoEvalResults(roughVideoScores, 'Rough_L')
    
    def computeCiderScore(self):
        score, ciderVideoScores = Cider().compute_score(self.gtc_tokens, self.pc_tokens)

        self.evalResults['Cider'] = score
        self.setVideoEvalResults(ciderVideoScores, 'Cider')
    
    def evaluate(self):
        self.computeBleuScore()
        # self.computeMeteorScore() #Gives error in Windows
        self.computeRoughScore()
        self.computeCiderScore()

    def setVideoEvalResults(self, videoScores, methodName):
        for videoId, score in zip(self.videoIds, videoScores):
            if not videoId in self.evalVideoResult:
                self.evalVideoResult[videoId] = dict()
            self.evalVideoResult[videoId][methodName] = score

    def getOverallScores(self):
        return self.evalResults
    
    def getVideowiseScores(self):
        return self.evalVideoResult

if __name__ == "__main__":
    import csv
    import config
    csv_file = config.TEST_SET_PREDICTION_CSV_FILE
    predicted_dataset = dict()
    with open(csv_file, newline='', encoding="utf8") as csvFile:
        csvDataReader = csv.DictReader(csvFile, delimiter='|')
        for row in csvDataReader:
            predicted_dataset[row['Video Id']] = [row['Original caption'], row['Predicted caption']]

    # print(len(predicted_dataset))
    # print(predicted_dataset['Hd-NeIhbYGc_43_48'])
    cocoEval = COCOCaptionEvaluator(predicted_dataset)
    cocoEval.evaluate()
    results = cocoEval.getOverallScores()
    print(results)