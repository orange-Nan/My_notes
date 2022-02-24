#导入库包
import pandas
from pandas.plotting import scatter_matrix
import matplotlib.pyplot as plt
from sklearn import model_selection
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

#导入数据集（鸢尾花数据集，分类）
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']
dataset = pandas.read_csv(url, names=names)

#数据集基本信息
#print(dataset.shape)   #shape数据集维度
#print(dataset.head(20))   #数据集前20行
#print(dataset.describe())   #descriptions统计摘要（数量、平均值、最大值、最小值等）
#print(dataset.groupby('class').size())  #class distribution每一个类别（每一种鸢尾花）行的数量

#数据可视化
#dataset.plot(kind='box', subplots=True, layout=(2,2), sharex=False, sharey=False)  #为每一个变量创建箱线图
#dataset.hist() #为每一个变量创建直方图（其中有2个变量呈高斯分布（正态分布））
#scatter_matrix(dataset)  #全部属性对的散点图，注意一些属性对呈对角线分布，这显示了它们之间有高度关联性以及可预测的关系
#plt.show()  #显示图片（如果想同时呈现以上三种图片，可以创建子图，在子图中绘制）

#评估算法
#1.创建验证集（80%训练，20%验证）
#得到的 X_train 和 Y_train 里的训练数据用于准备模型
array = dataset.values
X = array[:,0:4]
Y = array[:,4]
validation_size = 0.20
seed = 7
X_train, X_validation, Y_train, Y_validation = model_selection.train_test_split(X, Y, test_size=validation_size, random_state=seed)
#2.使用十折交叉验证法测试模型的准确度
scoring = 'accuracy'
#3.搭建和评估模型
models = []
models.append(('LR', LogisticRegression()))   #逻辑回归
models.append(('LDA', LinearDiscriminantAnalysis()))  #线性判别分析
models.append(('KNN', KNeighborsClassifier()))  #K近邻算法
models.append(('CART', DecisionTreeClassifier()))  #决策树
models.append(('NB', GaussianNB()))
models.append(('SVM', SVC()))
# evaluate each model in turn
results = []
names = []
for name, model in models:
    kfold = model_selection.KFold(n_splits=10,shuffle=True,random_state=seed)
    cv_results = model_selection.cross_val_score(model, X_train, Y_train, cv=kfold, scoring=scoring)
    results.append(cv_results)
    names.append(name)
    msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
    print(msg)
#将模型评估结果用图形表示出来，比较每个模型的跨度和平均准确度
#fig = plt.figure()
#fig.suptitle('Algorithm Comparison')
#ax = fig.add_subplot(111)
#plt.boxplot(results)
#ax.set_xticklabels(names)
#plt.show()

#直接在验证集上运行 KNN 算法
#将结果总结为一个最终准确率分值，一个混淆矩阵和一个分类报告
knn = KNeighborsClassifier()
knn.fit(X_train, Y_train)
predictions = knn.predict(X_validation)
print(accuracy_score(Y_validation, predictions))
print(confusion_matrix(Y_validation, predictions))
print(classification_report(Y_validation, predictions))
