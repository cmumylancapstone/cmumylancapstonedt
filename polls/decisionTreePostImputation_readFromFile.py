from builtins import print

import numpy as np
import pandas as pd
from sklearn.cross_validation import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn import tree
import matplotlib.pyplot as plt
import subprocess



from sklearn.externals.six import StringIO
from sklearn.tree import export_graphviz
import pydotplus




# read in data
dataDF = pd.read_csv('inputData.csv', sep= ',')

# PLOTTING TREE ONLY FOR A STATE
state = "DC"


# # BLOCK USED TO PRINT OUT PNGs FOR ALL STATES
# x = 1
# tempStateList = list(set(dataDF["State"]))
# print(len(tempStateList))
# for sateName in tempStateList:
#     state = sateName
#
#     dataDF = pd.read_csv('inputData.csv', sep=',')
#     if state == "DC":
#         break
#
#     print(x,"subsettng state:", state)
#     x += 1









# state = "national" # a placeholder for state. DONT DELETE THIS


#

# taking only the subset of data as per the above filter

dataDF = dataDF[dataDF['State'] == state]







# making sure that the data types of all the columns now is numeric
dataDF = dataDF.apply(pd.to_numeric, errors='ignore')


# replacing the readmissionrate by a bucketed column called Rate
dataDF['Rate'] = pd.qcut(dataDF["Hospital Readmission Rate"], 6)

classNames = list(set(dataDF['Rate']))
for i in range(len(classNames)):
    aClassName = classNames[i]
    aClassName = aClassName[1: (len(aClassName) - 1)]
    classNames[i] = aClassName


classNames.sort()
# print(classNames)


# deleting the state-county etc columns
dataDF = dataDF.drop(["Year", "State", "County"], 1)


# print(dataDF.shape)

# removing this ["Hospital Readmission Rate"] clumn now - we will onyl use the newly generated rate column
dataDF = dataDF.drop("Hospital Readmission Rate", 1)
#
# print(dataDF.shape)
#
# print(dataDF.columns.values)
#
# print(dataDF.iloc[0:10])



# seaprating the X's and the Y column
X = dataDF.drop("Rate", 1)
Y = pd.DataFrame(dataDF["Rate"])

# print(X.shape)
# print(Y.shape)


# taking out the feature names ofr annotating the decision tree later
featureNames = X.columns.values

# print(featureNames)
#
# print(X.head())

# fitting the decision tree
clf_gini = DecisionTreeClassifier(criterion = "entropy", random_state = 100, max_depth=4)
clf_gini.fit(X, Y)





# writing file to pdf
dot_data = StringIO()
export_graphviz(clf_gini, out_file=dot_data,
                filled=False, rounded=True,
                special_characters=True, feature_names = featureNames, impurity = False, class_names = classNames)

graph = pydotplus.graph_from_dot_data(dot_data.getvalue())

# graph.write_pdf(state+"_tree.pdf")

graph.write_png("1_"+state+"_tree.png")

