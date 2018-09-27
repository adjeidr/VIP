import pandas
import sklearn
import numpy as np
import ast
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import sklearn.preprocessing as pre
from sklearn.feature_extraction import DictVectorizer
from sklearn.datasets import load_breast_cancer


def dataChanger(dataArr):
    hold = list()
    for each in dataArr:
        hold.append(ast.literal_eval(each))
    return hold

file = 'citalopram_train.csv'
df = pandas.read_csv(file, names = ["data","target"])
print(df.shape)
data = np.array(df["data"])
target = np.array(df["target"])

data = dataChanger(data)

v = DictVectorizer(sparse=False)
v.fit_transform(data)

enc = pre.LabelEncoder()
print(enc.fit_transform(target))


# cancer = load_breast_cancer()
# x_train, x_test, y_train, y_test = train_test_split(v, enc, random_state = 0)
#
# forest = RandomForestClassifier(n_estimators=100, random_state=0)
# forest.fit(x_train, y_train)
#
# print("Accuracy of training the data set {:.3f}".format(forest.score(x_train, y_train)))
# print("Accuracy of of the test subset {:.3f}".format(forest.score(x_test,y_test)))
