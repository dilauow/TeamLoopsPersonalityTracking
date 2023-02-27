from src.TraitCalculator import TraitCalculator

X_test = [0,0,0,0,0]
X_train =[0,0,0,0,0]

y_test =[0,0,0,0,0]
y_train =[0,0,0,0,0]

tc = TraitCalculator()
tc.loadData()
print(len(tc.sentences))
print(len(tc.labels[1]))

clfs = []


for i in range(5):
    X_tr, X_te, y_tr, y_te = tc.splitDataset(tc.sentences[i*2000:((i+1)*1900)+2000],tc.labels[i][i*2000:((i+1)*1900)+2000])
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
    paddedUserQuery = tc.preprocessor(["I'm not afraid to admit when I don't know something and to ask for help or clarification when needed","I'm always willing to listen to feedback and make adjustments to my approach. This has helped me grow and improve over time","I'm not open to working in a fast-paced environment that requires quick decisions",sentence])
    for clf in clfs:
        print(clf.predict(paddedUserQuery))



