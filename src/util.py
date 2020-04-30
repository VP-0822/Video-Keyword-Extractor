import os

def writeArrayToFile(filepath, arrayToWrite):
    with open(filepath, "w") as txt_file:
        for line in arrayToWrite:
            txt_file.write(line + "\n")

def readArrayFromFile(filepath):
    return_array = list()
    with open(filepath, 'r+') as text_file:
        for line in text_file.readlines():
            return_array.append(line)
    return_array = [x.strip() for x in return_array]
    return return_array

def writeLossAndAccuracyToFile(filepath, noOfEpochs, trainingLoss, validationLoss, trainingAcc, validationAcc):
    with open(filepath, "w") as outfile:
        outfile.write('Number of Epochs: ' + str(noOfEpochs))
        outfile.write('\n\n')

        outfile.write('Training Loss:\n')
        outfile.write(", ".join(str(x) for x in trainingLoss))
        outfile.write('\n\n')

        outfile.write('Validation Loss:\n')
        outfile.write(", ".join(str(x) for x in validationLoss))
        outfile.write('\n\n')

        outfile.write('Training Accuracy:\n')
        outfile.write(", ".join(str(x) for x in trainingAcc))
        outfile.write('\n\n')

        outfile.write('Validation Accuracy:\n')
        outfile.write(", ".join(str(x) for x in validationAcc))
        outfile.write('\n\n')
    print('Training loss and accuracy saved to a file.')