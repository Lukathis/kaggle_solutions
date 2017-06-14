import pandas as pd

import warnings
warnings.filterwarnings("ignore")

import matplotlib.pyplot as plt

import seaborn as sns
#
# read in the iris dataset
iris = pd.read_csv("/Users/Chi/PycharmProjects/kaggle_solutions/iris/data/iris.csv")
iris.head()

print(iris["Species"].value_counts())
iris.plot(kind="scatter", x="SepalLengthCm", y="SepalWidthCm")
sns.jointplot(x="SepalLengthCm", y="SepalWidthCm", data=iris, size=5)
sns.FacetGrid(iris, hue="Species", size=5).map(plt.scatter, "SepalLengthCm", "SepalWidthCm").add_legend()
sns.boxplot(x="Species", y="PetalLengthCm", data=iris)
ax = sns.boxplot(x="Species", y="PetalLengthCm", data=iris)
ax = sns.stripplot(x="Species", y="PetalLengthCm", data=iris, jitter=True, edgecolor="gray")

plt.show()