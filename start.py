import pandas
import sklearn
import numpy as np
import ast
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import sklearn.preprocessing as pre
from sklearn.datasets import load_breast_cancer

def dataChanger(dataArr):
    hold = list()
    keys = list()
    for each in dataArr:
        hold.append(ast.literal_eval(each))
    for each in hold:
        keys.append([i for i in each.keys()])
    return keys

file = 'citalopram_train.csv'
df = pandas.read_csv(file, names = ["data","target"])
print(df.shape)
data = np.array(df["data"])
target = np.array(df["target"])

words = dataChanger(data)
print(words)
# features = pandas.get_dummies(df)
# print(features)
# print(features.describe())

le1 = pre.LabelEncoder()
le1.fit(words)
print(le1.classes_)
transform1 = le1.transform(words)

le2 = pre.LabelEncoder()
le2.fit(target)
transform2 = le2.transform(target)
print(transform2)

cancer = load_breast_cancer()
print(type(cancer))
x_train, x_test, y_train, y_test = train_test_split(cancer.data, cancer.target, random_state = 0)

forest = RandomForestClassifier(n_estimators=100, random_state=0)
forest.fit(x_train, y_train)

print("Accuracy of training the data set {:.3f}".format(forest.score(x_train, y_train)))
print("Accuracy of of the test subset {:.3f}".format(forest.score(x_test,y_test)))
