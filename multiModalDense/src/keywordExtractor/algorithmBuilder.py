import pke

def createInstanceOfPkeAlgo():
    switcher = {
        1: topicRank,
        2: topicalPageRank,
        3: textRank,
        4: positionRank,
        5: multipartiteRank,
        6: singleRank,
        7: yAKE,
        8: kPMiner,
        9: tfIdf,
        10: seq2Seq,
        11: topicCoRank,
        12: kea,
        13: wignus
    }

    algorithmIndex = getAlgorithmDict()[algorithm]
    func = switcher.get(algorithmIndex, lambda: "Invalid algorithm")
    keywordExtractorAlgo = func()
    return keywordExtractorAlgo

def topicRank():
    return pke.unsupervised.TopicRank()

def topicalPageRank():
    return pke.unsupervised.TopicalPageRank()

def textRank():
    return pke.unsupervised.TextRank()

def positionRank():
    return pke.unsupervised.PositionRank()

def multipartiteRank():
    return pke.unsupervised.MultipartiteRank()

def singleRank():
    return pke.unsupervised.SingleRank()

def yAKE():
    return pke.unsupervised.YAKE()

def kPMiner():
    return pke.unsupervised.KPMiner()

def tfIdf():
    return pke.unsupervised.TfIdf()

def seq2Seq():
    return pke.supervised.Seq2Seq()

def topicCoRank():
    return pke.supervised.TopicCoRank()

def kea():
    return pke.supervised.Kea()

def wignus():
    return pke.supervised.WINGNUS()

def getAlgorithmDict():
    algorithmDict = {}
    algorithmDict['TopicRank'] = 1
    algorithmDict['TopicalPageRank'] = 2
    algorithmDict['TextRank'] = 3
    algorithmDict['PositionRank'] = 4
    algorithmDict['MultipartiteRank'] = 5
    algorithmDict['SingleRank'] = 6
    algorithmDict['YAKE'] = 7
    algorithmDict['KPMiner'] = 8
    algorithmDict['TfIdf'] = 9
    algorithmDict['Seq2Seq'] = 10
    algorithmDict['TopicCoRank'] = 11
    algorithmDict['Kea'] = 12
    algorithmDict['Wignus'] = 13
    return algorithmDict