import math

class adaboost:

    def buildAdaBoost(self, examples, features):

        # number of hypothesis in the ensemble
        val_K = 10
        wList = []  #vector of N example weights
        hk = []  #vector of K hypothesis
        index_k = []  #vector of K hypothesis weights

        hk = self.L(examples, wList)
        hypList = []

        for index_k in range(val_K):
            listStump = []
            val_a = ''
            minErrorValue = math.inf
            val_y = []


            for j_val in range(len(features)):

                diffVal = 0
                correction = []

                for l_val in range(examples.shape[0]):
                    if (examples.values[l_val][j_val] is hk[l_val]):
                        correction.append(l_val)
                    else:
                        diffVal += wList[l_val]

                    [minErrorValue, val_y, val_a] = self.errorCheck(diffVal, minErrorValue, correction, features, j_val, val_y, val_a)

            for ind in val_y:
                wList[ind] *= minErrorValue / (1 - minErrorValue)

            wList = self.calculateNormalWeight(wList)
            votes_z = math.log((1 - minErrorValue) / (diffVal), 2)

            listStump.append(val_a)
            listStump.append(votes_z)
            hypList.append(listStump)
            features.remove(val_a)
            features.append(val_a)

        return hypList

    def L(self, examples, wList):
        english = 'en'
        for index in range(examples.shape[0]):
            wList.append(1 / examples.shape[0])
        inputIndexList = list(examples.index.values)
        finOut = []

        for l in inputIndexList:
            if l == english:
                finOut.append(False)
            else:finOut.append(True)
        return finOut

    def calculateNormalWeight(self, inputWeights):

        weightSum = sum(inputWeights)

        weightValueList = []
        for obj in inputWeights:
            weightValueList.append(obj / weightSum)
        return weightValueList

    def errorCheck(self, error, errorMin, cor, attributes, j_val, y, a):
        if error < errorMin:
            errorMin = error
            y = cor
            a = attributes[j_val]
        return [errorMin, y, a]


def buildAda(data, attribute):
    adaObj = adaboost()
    adaVal = adaObj.buildAdaBoost(data, attribute)
    return adaVal