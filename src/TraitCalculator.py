import numpy as np

from src.Preproccessor import TextPreprocessor
from src.dataHandler import TrainingData
from src.randomForest import TraitModel


class TraitCalculator:
    def __init__(self):
        self.T_Processor = TextPreprocessor()
        self.MlModel = TraitModel()
        self.labels =[]
        self.sentences = []
        self.allTraitLabel=[]


    def loadData(self):
        data = TrainingData()
        data.importToModel()
        sentences = np.array(data.sentences)
        self.labels = [data.openness,data.consciensiosness,data.extraversion,data.agreebleness,data.nurotisicm]
        self.sentences = data.sentences
        self.allTraitLabel = data.allTraitLabel
        return sentences
    def splitDataset(self,features,label):
        X_train, X_test, y_train, y_test = self.MlModel.test_train_split(0.8,features,label)
        return X_train, X_test, y_train, y_test

    def preprocessor(self,features):
        preprocessedText = self.T_Processor.feedPreprocessorAnArray(features)
        paddedArr = self.T_Processor.sequencePadding(preprocessedText)
        return paddedArr

    def TrainModel(self,features,label):
        clf = self.MlModel.randomForest(features,label)
        return clf

    def testClassifier(self,features,label,clf):

        self.MlModel.test_classifiers(features,label,clf)






