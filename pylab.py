import math
import pickle
import re
import pandas as panData

from DTree import DTree
from Attribute import sentenceFeature
from AdaBoost import buildAda

def pluralityValue(parent_examples):
    return max(list(parent_examples.index.values), key=list(parent_examples.index.values).count)

def findClassification(examples):

    classifierSame = False
    classifierVal = 'en'

    englishValues, dutchValues = getCountValues(list(examples.index.values))

    if englishValues == 0:
        classifierSame = True
    if dutchValues == 0:
        classifierSame = True
    if dutchValues > englishValues:
        classifierVal = 'nl'
    return [classifierSame, classifierVal]


def checkForProb(P_x, P_y):

    if P_x == 0 or P_x == 1:
        return True
    if P_y == 1 or P_y == 0:
        return True
    return False


def calcEntropy(val_x, val_y):
    if (val_x + val_y) < 0 or (val_x + val_y) == 0:
        return 0
    else:
        P_x = val_x / (val_x + val_y)
        P_y = val_y / (val_x + val_y)

    if checkForProb(P_x, P_y):
        return 0

    entropyVal = -(P_x * math.log(P_x, 2)) - (P_y * math.log(P_y, 2))
    # (P_x * (1/(math.log(P_x, 2)))) + (P_y * (1/(math.log(P_y, 2))))
    return entropyVal


def getCountValues(inputList):
    return [inputList.count('en'), inputList.count('nl')]


def getTrueValCount(index, true_A, english_true, dutch_true):
    englishLang = 'en'
    true_A += 1
    if index == englishLang:
        english_true = english_true + 1
    else:
        dutch_true = dutch_true + 1
    return [true_A, english_true, dutch_true]


def getFalseValCount(index, false_A, english_false, dutch_false):
    dutchLang = 'nl'
    false_A += 1
    if index == dutchLang:
        dutch_false += 1
    else:
        english_false += 1
    return [false_A, english_false, dutch_false]

def checkForGain(A, gainA, gainCount, attribute):
    if gainA > gainCount and not gainA == 0:
        A = attribute
        gainCount = gainA
    return A, gainCount

def Importance(attributes, examples):

    inputList = list(examples.index.values)
    countValues = getCountValues(inputList)

    entropy_B = calcEntropy(countValues[1], countValues[0])
    A_val = ''
    gainCount = 0

    for attribute in attributes:

        trueCountValues = []
        falseCountValues = []

        true_A, false_A, dutch_true, english_true, dutch_false, english_false = 0, 0, 0, 0, 0, 0

        for row in range(examples.shape[0] - 1):

            index = inputList[row]

            if (examples[attribute][row] == True):
                trueCountValues = getTrueValCount(index, true_A, english_true, dutch_true)
                true_A = trueCountValues[0]
                english_true = trueCountValues[1]
                dutch_true = trueCountValues[2]

            if not (examples[attribute][row] == True):
                falseCountValues = getFalseValCount(index, false_A, english_false, dutch_false)
                false_A = falseCountValues[0]
                english_false = falseCountValues[1]
                dutch_false = falseCountValues[2]

        remainder = ((true_A / examples.shape[0]) * calcEntropy(dutch_true, english_true)) + ((false_A / examples.shape[0]) * calcEntropy(dutch_false, english_false))
        gainA = entropy_B - remainder

        A_val, gainCount = checkForGain(A_val, gainA, gainCount, attribute)

    return A_val

def checkIfSameForClassifier(outputVal):
    for _ in outputVal:
        continue
    return outputVal[0]

def dTreeLearning(inputExamples, attributeList, parentExamples):

    outputVal = findClassification(inputExamples)

    if inputExamples.empty:
        return DTree(pluralityValue(parentExamples))

    elif checkIfSameForClassifier(outputVal):
        return DTree(outputVal[1])

    elif attributeList == False:
        return DTree(pluralityValue(inputExamples))

    else:
        A_val = Importance(attributeList, inputExamples)

        return DTree(A_val, dTreeLearning(inputExamples.loc[inputExamples[A_val] == False], attributeList.remove(A_val), inputExamples),
                     dTreeLearning(inputExamples.loc[inputExamples[A_val] == True], attributeList, inputExamples))

def train(examples, hypothesisOut, learningType):

    attributes = ['common_dutch', 'common_english', 'qWord', 'ooWord', 'eenWord', 'aaWord', 'ijWord', 'eeWord',
                  'deWord', 'enWord', 'vanWord']
    languageInFile = []
    attributeBoolFeat = []
    trainFile = open(examples, encoding="UTF-8")

    for line in trainFile:
        languageInFile.append(line[:2])
        sentence = re.sub("[!@#$%^&*()[]{};:,./<>?\|`~-=_+]", " ", line[3:])
        attributeBoolFeat.append(sentenceFeature(sentence))

    finData = panData.DataFrame(attributeBoolFeat, columns=attributes, index=languageInFile)

    if learningType == 'ada':
        tree = buildAda(finData, attributes)
    else:
        tree = dTreeLearning(finData, attributes, finData)

    pickle.dump(tree, open(hypothesisOut, 'wb'))

def buildDTree(treeVal, hyp, lang):

    if hyp.left is None or hyp.right is None:
        return hyp.value
    elif not treeVal[hyp.value][lang] == True:
        return buildDTree(treeVal, hyp.left, lang)
    else: return buildDTree(treeVal, hyp.right, lang)

def decTreePrint(file, finOutput):

    print("Prediction using decision tree:\n")
    with open(file, encoding="UTF-8") as f:
        for line in f:
            for val in finOutput:
                print(val)
                finOutput.remove(val)
                break

def predictForDTree(h, file):

    finOutput = []
    input_file = open(file, 'r')

    attributes = ['common_dutch', 'common_english', 'qWord', 'ooWord', 'eenWord', 'aaWord', 'ijWord', 'eeWord',
                  'deWord', 'enWord', 'vanWord']
    boolLang = []
    valForData = []

    for line in input_file:
        boolLang.append(line[:2])
        sentence = re.sub("[!@#$%^&*()[]{};:,./<>?\|`~-=_+]", " ", line[0:])
        valForData.append(sentenceFeature(sentence))

    finData = panData.DataFrame(valForData, columns=attributes)

    for val in range(finData.shape[0]):
        treeVal = buildDTree(finData, h, val)
        finOutput.append(treeVal)
    decTreePrint(file, finOutput)

def adaBPrint(file, finOutput):

    print("Prediction using AdaBoost:\n")
    with open(file, encoding="UTF-8") as f:
        for line in f:
            for val in finOutput:
                print(val)
                finOutput.remove(val)
                break


def findClassifier(data, hypo, indexVal, classifierValue):
    if data[hypo[0]][indexVal]:
        classifierValue += hypo[1]
    else:
        classifierValue -= hypo[1]
    return classifierValue


def predictForAdaB(indexes, file):
    finOutput = []
    attributes = ['common_dutch', 'common_english', 'qWord', 'ooWord', 'eenWord', 'aaWord', 'ijWord', 'eeWord',
                  'deWord', 'enWord', 'vanWord']

    boolLang = []
    attributeVal = []

    with open(file, 'r') as inputFile:
        for l in inputFile:
            boolLang.append(l[:2])
            sentence = re.sub("[!@#$%^&*()[]{};:,./<>?\|`~-=_+]", " ", l[3:])
            attributeVal.append(sentenceFeature(sentence))

    finData = panData.DataFrame(attributeVal, columns=attributes)

    for indexVal in range(finData.shape[0]):
        classifierValue = 0

        for index_val in indexes:
            classifierValue = findClassifier(finData, index_val, indexVal, classifierValue)

        if classifierValue < 0: finOutput.append('nl')
        else:
            finOutput.append('en')

    adaBPrint(file, finOutput)

def trainModel():
    trainFile = input("enter the training file <trainFile>: \n")
    hypFile = input("enter hypothesis file <hypFile>:\n")
    learnType = input("Select the learning type. 'dt' or 'ada' <learning-type>?\n")

    attributes = ['common_dutch', 'common_english', 'qWord', 'ooWord', 'eenWord', 'aaWord', 'ijWord', 'eeWord',
                  'deWord', 'enWord', 'vanWord']
    language = []
    boolFeat = []

    specialChar = "!()-?&[]\,\”\“<>.{};:"
    with open(trainFile, encoding="UTF-8") as trainFile:
        for line in trainFile:
            language.append(line[:2])
            sentence = line[3:].lower()
            for charVal in specialChar:
                if charVal in sentence:
                    sentence = sentence.replace(charVal, '')
            boolFeat.append(sentenceFeature(sentence))

    data = panData.DataFrame(boolFeat, index=language, columns=attributes)

    if learnType == 'dt':
        currentDtree = dTreeLearning(data, attributes, data)
    elif learnType == 'ada':
        currentDtree = buildAda(data, attributes)
    pickle.dump(currentDtree, open(hypFile, 'wb'))
    print("Model Training...")

def predictModel():
    hypothesis = input("enter the hypothesis file path\n")
    file = input("enter testing file path\n")

    h = pickle.load(open(hypothesis, 'rb'))
    if isinstance(h, DTree):
        predictForDTree(h, file)
    else:
        predictForAdaB(h, file)

def main():
    userInput = input("'train' or 'predict' model?\n")
    if userInput == 'predict':
        predictModel()
    else:
        trainModel()

if __name__ == '__main__':
    main()
