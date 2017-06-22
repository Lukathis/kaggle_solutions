import pandas as pd

import warnings
warnings.filterwarnings("ignore")

import matplotlib.pyplot as plt

import seaborn as sns
#
# read in the iris dataset
iris = pd.read_csv("/Users/Chi/PycharmProjects/kaggle_solutions/iris/data/iris.csv")
print(iris.head())

''' 查看iris以及子集 '''
# print(iris)
# print(iris['SepalLengthCm'])
# print(iris[['SepalLengthCm', 'SepalWidthCm']])

print(iris.describe())

''' 绘制除了ID列之外其他列的范围分布boxplot '''
# iris_features = iris[['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']];
# iris_features.boxplot()

''' 绘制散点图 '''
plt.scatter(iris['PetalWidthCm'], iris['PetalLengthCm'], alpha=1.0, color='k')
plt.xlabel('Petal Width')
plt.ylabel('Petal Length')
plt.show()

''' 绘制直方图 '''
plt.hist(iris['PetalWidthCm'], bins=20)
plt.xlabel('petal width distribution')


# print(iris["Species"].value_counts())
# iris.plot(kind="scatter", x="SepalLengthCm", y="SepalWidthCm")
# sns.jointplot(x="SepalLengthCm", y="SepalWidthCm", data=iris, size=5)
# sns.FacetGrid(iris, hue="Species", size=5).map(plt.scatter, "SepalLengthCm", "SepalWidthCm").add_legend()
# sns.boxplot(x="Species", y="PetalLengthCm", data=iris)
# ax = sns.boxplot(x="Species", y="PetalLengthCm", data=iris)
# ax = sns.stripplot(x="Species", y="PetalLengthCm", data=iris, jitter=True, edgecolor="gray")
#
plt.show()