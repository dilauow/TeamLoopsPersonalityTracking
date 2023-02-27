from src.TraitCalculator import TraitCalculator

X_test = [0,0,0,0,0]
X_train =[0,0,0,0,0]

y_test =[0,0,0,0,0]
y_train =[0,0,0,0,0]

tc = TraitCalculator()
tc.loadData()
print(len(tc.sentences))
print(len(tc.labels[3]))

clfs = []


for i in range(5):
    X_tr, X_te, y_tr, y_te = tc.splitDataset(tc.sentences,tc.labels[i])
    X_test[i] = X_te
    X_train[i] = X_tr
    y_test[i] = y_te
    y_train[i] =y_tr
    print("reacheed en")


for i in range(5):
    paddedArr = tc.preprocessor(X_train[i])
    clf = tc.TrainModel(paddedArr,y_train[i])
    clfs.append(clf)
    paddedTest = tc.preprocessor(X_test[i])
    tc.testClassifier(paddedTest,y_test[i],clf)


userBreak = "True"

while userBreak!="False":
    sentence = input("Enter your sentence")
    userBreak = input("ENter Flase to break")
    paddedUserQuery = tc.preprocessor([sentence])
    for clf in clfs:
        print(clf.predict(paddedUserQuery))



