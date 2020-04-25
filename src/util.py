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