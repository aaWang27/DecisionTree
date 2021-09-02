import matplotlib as mpl
from sklearn import tree
from sklearn.model_selection import cross_val_score
import pandas as pd
import numpy as np

mpl.use('TkAgg')

df_train = pd.read_csv('train.csv')
print(df_train.head())

x = df_train[['age', 'gender', 'height', 'weight', 'ap_hi', 'ap_lo', 'cholesterol', 'gluc', 'smoke', 'alco', 'active']]
y = df_train[['cardio']]

clf = tree.DecisionTreeClassifier(max_depth=8, min_samples_split=5)
clf = clf.fit(x, y)

df_test = pd.read_csv("test.csv")
x_test = np.array(df_test[['age', 'gender', 'height', 'weight', 'ap_hi', 'ap_lo', 'cholesterol', 'gluc', 'smoke', 'alco', 'active']])

y_predict = clf.predict(x_test)

with open('SKCardiovascularPredictions.csv', 'w') as f:
    f.write("Patient #,classification\n")
    for i in range(len(y_predict)):
        if (y_predict[i] == 0):
            classification = 'None'
        else:
            classification = 'Cardio'
        f.write("Patient " + str(i + 1) + "," + str(classification) + "\n")

tree.plot_tree(clf)
