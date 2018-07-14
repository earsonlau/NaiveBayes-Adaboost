# -*- coding: utf-8 -*-
"""
Created on Sat Apr 28 23:44:37 2018

@author: earson
"""

#libsvm格式的数据我很不喜欢，于是把它读成矩阵

#读取数据
import sys
import random
import numpy as np
import os
from collections import Counter

def data_generator(filename):#数据的读取
    """
    :param filename:
    :param batch_size:
    :return:
    """
    
    myfile = open(filename) #打开文件
    batch_size = len(myfile.readlines()) #通过计算读取文件有多少行得到数据行数
    for line in open(filename):#对打开的文本文件的每一行的字符按空格和换行符分开
        line=line.strip("\n")
        data=line.split(" ")
    feature_size=(len(data))#计算data的长度得到特征的个数
    labels = np.zeros(batch_size)
    rets = np.empty(shape=[batch_size, feature_size])#存储数据行数和特征个数
    i = 0
    for line in open(filename, "r"):#读取libsvm格式的文件
        data = line.split(" ")
        label = (data[0])
        ids = []
        values = []
        for fea in data[1:]:
            id, value = fea.split(":")
            if int(id) > feature_size - 1:
                break
            ids.append(int(id))
            values.append(value)
            
        ret = np.zeros([1, feature_size])
        for (index, d) in zip(ids, values):
            ret[0][index] = d
        labels[i] = int(label)#每一个lable都设为整型
        rets[i] = ret
        i += 1
        if i > batch_size - 1:
            i = 0
            yield labels, rets[0:, 1:feature_size]

def separatebyclass1(dataset):#在寻找最佳特征的时候，需要把数据按类分一下
    separated={}#创建一个字典
    for i in range(len(dataset)):
        vector=(dataset[i])
        if vector[0] not in separated:
            separated[vector[0]]=[]
        separated[vector[0]].append(vector[1:])#字典的key是vector[0]，value是vector[1:]
    return separated

def separatebyclass2(dataset):#在得知最佳特征后，也需要把数据按类分一下
    separated={}
    for i in range(len(dataset)):
        vector=(dataset[i])
        if vector[0] not in separated:
            separated[vector[0]]=[]
        separated[vector[0]].append(vector[1]) #字典的key是vector[0]，value是vector[1]
    return separated

def findbestfeature(databyclass):#寻找最佳的特征,databyclass是按所属类分好的数据
    temp=[]
    var=[]#存储
    temp2=[]
    
    for j in range(1,len(databyclass)):#对于每一类
        varoffeature=[]
        for k in range(len(databyclass[j][0])):#对于该类的每一个特征
            for i in range(len(databyclass[j])):#对于类内的每一个样本
                #for j in range(len(databyclass[1][0]))
                temp.append((databyclass[j])[i][k])#把样本对应特征的值放在一个list中
            varoffeature.append(np.var(temp))#计算特定类特定特征的所有样本值的方差
        var.append(varoffeature)  #存在一个list中

    for i in range(len(var)):
        temp2.append(var[i].index(min(var[i]))+1)#找出每一类中方差最小的特征
    
    var_counts = Counter(temp2)
    feature_num=var_counts.most_common(1)[0][0]
    
    return feature_num#找出出现方差最小最多次的特征


def load_data(train_data, test_data, D):  # 读取数据
    train_data_generator = data_generator(train_data)  # 数据集的长度
    train_data_labels, train_data_features = train_data_generator.__next__()
    train_data_lab = np.transpose([train_data_labels])  # transpose,把标签弄成一列并加到feature的前面
    train_data_total = np.append(train_data_lab, train_data_features, axis=1)
    train_data_total_with_weight = make_data_have_weight(train_data_total, D)
    train_databyclass_with_weight = separatebyclass1(train_data_total_with_weight)
    feature_num = findbestfeature(train_databyclass_with_weight)
    train_set1 = train_data_total_with_weight[:, (0, feature_num)]

    test_data_generator = data_generator(test_data)  # 数据集的长度
    test_data_labels, test_data_features = test_data_generator.__next__()
    test_data_lab = np.transpose([test_data_labels])  # transpose,把标签弄成一列并加到feature的前面
    test_data_total = np.append(test_data_lab, test_data_features, axis=1)
    test_databyclass = separatebyclass1(test_data_total)

    test_set = test_data_total[:, (0, feature_num)]
    return train_set1, test_set, feature_num, train_databyclass_with_weight, train_data_total_with_weight


def make_data_have_weight(train_data_total, D):  # 给数据添加权重
    train_data_total_with_weight = np.c_[train_data_total, D]
    train_data_total_with_weight = np.squeeze(np.asarray(train_data_total_with_weight))
    return train_data_total_with_weight


def countPC(sep,m):#计算P(Ci)
    PC=[]
    for i in range(1,(len(sep)+1)):
        PC.append(len(sep[i])/m)#计算各个类在数据中的频率
    return dict(zip(list(sep.keys()),PC))#合成字典

def countPXC(sepi):#计算P(X|Ci)
    PXC=[]
    frequency=dict(Counter(sepi))
    for i in range(int(min(frequency.keys())),int(max(frequency.keys())+1)):
        PXC.append(frequency[i]/len(sepi))#计算特定类内各个样本出现的频率
    
    return dict(zip(list(frequency.keys()),PXC))#合成字典

# def calculatefrequency(sep):#计算每一类中元素们出现的频率
#     for i in range(1,len(sep)+1):
#         print('Frequency of Class {0}:{1}\n Frequncy of element in this Class: {2}'.format(i,countPC(sep)[i],countPXC(sep[i])))
    
#def model_trainning(sep_with_weight,sep):#做预测
#    
#    tags=[]
#    for j in range(1,len(Counter(sep[1]))+1):#求出每一组特征值最大概率是属于哪一类
#        for i in range(1,len(sep)+1):#对每一个特征值（此处特征值是1*1的）
#            if i==1:
#                MAX=countPC(sep_with_weight)[i]*countPXC(sep[i])[j]#计算概率
#                tag=i
#            elif countPC(sep_with_weight)[i]*countPXC(sep[i])[j] > MAX:#求最大概率
#                MAX=countPC(sep_with_weight)[i]*countPXC(sep[i])[j]
#                tag=i
#            else :
#                continue
#        tags.append(tag)
#    train_result=tags#把模型训练后的结果保存为train_result，为一个列表（list），长度为特征允许的值的最大值
#    return train_result
def model_trainning(sep,m):#做预测
    train_result=[]
    row_number=int(max(max(sep.values())))
    matrix = np.zeros((row_number, len(sep)))
    for i in range(1,len(sep)+1):#对每一个特征值（此处特征值是1*1的）
        for j in range(int(min(Counter(sep[i]).keys())),int(max(Counter(sep[i]).keys()))+1):#求出每一组特征值最大概率是属于哪一类
            matrix[j-1][i-1]+=countPC(sep,m)[i]*countPXC(sep[i])[j]#计算概率
    for k in range(row_number):
         train_result.append(np.argmax(matrix[k])+1)#把模型训练后的结果保存为train_result，为一个列表（list），长度为特征允许的值的最大值
    return train_result


def predict(test_set,train_result):#预测
    tags=train_result
    predictions=[]
    for k in range(len(test_set)):
        result = tags[int(test_set[k][1])-1]
        predictions.append(result)
    return predictions



# def confusion_matrix(test_set,prediction,sep):#输出混淆矩阵
#     row_number = int(max(max(sep.values())))
#     confusion_matrix=np.zeros((len(sep),len(sep)))#confusion_matrix[i][j]就是类i-1被识别为类j-1的个数
#     for i in range(len(test_set)):
#             confusion_matrix[int((test_set[:,0])[i])-1][(prediction[i])-1]+=1
#     return confusion_matrix
def confusion_matrix_for_adaboost(test_set,prediction,sep):#输出混淆矩阵
    row_number = int(max(max(sep.values())))
    confusion_matrix=np.zeros((len(sep),len(sep)))#confusion_matrix[i][j]就是类i-1被识别为类j-1的个数
    for i in range(len(test_set)):
            confusion_matrix[int((test_set[i]))-1][int(prediction[i])-1]+=1
    return confusion_matrix

def describe(confusion_matrix):

    x=0.000001
    L = np.tril(confusion_matrix, -1)
    U = np.triu(confusion_matrix, 1)
    D = np.mat(np.diag(np.diag(confusion_matrix)))
    print("Overall accuracy:",D.sum()/confusion_matrix.sum())
    TP = []
    FP = []
    FN = []
    TN = []

    for i in range(len(confusion_matrix[0])):
        TP.append(D[i, i])
        FP.append(U[i].sum() + L[i].sum())
        FN.append(L[:, i].sum() + U[:, i].sum())
        TN.append(D.sum() - D[i, i])

    TP = np.array(TP)
    FP = np.array(FP)
    FN = np.array(FN)
    TN = np.array(TN)
    print(TP, FP, FN, TN)
    accuracy = (TP + TN) / (TP + TN + FP + FN)
    specificity = (TN+x) / (TN + FP+x)
    precision = (TP+x) / (TP + FP+x)
    recall = (TP+x) / (TP + FN+x)
    F1score = 2 * precision * recall / (2 * precision + recall)
    F5score = 1.25 * precision * recall / (0.25 * precision + recall)
    F2score = 5 * precision * recall / (4 * precision + recall)
    for i in range(len(confusion_matrix[0])):
        print("The description:", "accuracy:",accuracy[i], "specificity:",specificity[i], "precision:",precision[i], "recall:",recall[i], "F1score:",F1score[i], "F5score:",F5score[i],
              "F2score:",F2score[i])
    return 1
def getdatalength(train_data):#得到数据集的长度

    train_data_generator = data_generator(train_data)  # 数据集的长度
    train_data_labels, train_data_features = train_data_generator.__next__()
    train_data_lab = np.transpose([train_data_labels])  # transpose,把标签弄成一列并加到feature的前面
    train_data_total = np.append(train_data_lab, train_data_features, axis=1)
    return len(train_data_total)

def get_number_with_weight(train_data_total_with_weight):#得到记录的序号和对应的权重
    
    number=[]
    weight=[]
    for i in range(len(train_data_total_with_weight)):
        number.append(i)
        weight.append(train_data_total_with_weight[i][-1])
#     number_with_weight=dict(zip(number,weight))
    return number,weight

def get_sample_by_weight(number,weight,error_count,train_data_total_with_weight):#通过权重挑选样本
    A=[]
    list = number
    weight1 = [x*(max(number)+1) for x in weight]
    """:param weight: list对应的权重序列
    :return:选取的值在原列表里的索引
    """
    for k in range(error_count):
        t = random.randint(0, int(round(sum(weight1))) - 1)
        for i, val in enumerate(weight1):
            t -= val
            if t < 0:
                a=i
                break
        A.append(a)
        
    number_of_samples=A
    samples=[]
    for i in number_of_samples:
        samples.append(train_data_total_with_weight[i])
    return np.array(samples)

def buildNaiveBayes(train_data,test_data,D):#构建朴素贝叶斯分类器

    train_set = load_data(train_data, test_data, D)[0]  # 导入训练集
    test_set = load_data(train_data, test_data, D)[1]  # 导入测试集
    m = getdatalength(train_data)
    feature_num = load_data(train_data, test_data, D)[2]  # 计算导入选定的特征是第几个
    train_databyclass_with_weight = load_data(train_data, test_data, D)[3]
    train_data_total_with_weight = load_data(train_data, test_data, D)[4]
    number=get_number_with_weight(train_data_total_with_weight)[0]
    weight=get_number_with_weight(train_data_total_with_weight)[1]
    samples=get_sample_by_weight(number, weight, len(train_set),train_data_total_with_weight)
    train_set=samples[:, (0, feature_num)]
    sep = separatebyclass2(train_set)  # 对特征进行简化
    #     #     calculatefrequency(sep)
    train_result = model_trainning(sep,m)  # 训练
    #print("Classify rule:\n",train_result)  # 打印训练结果
    class_labels=test_set[:, 0]
    prediction = np.array(predict(test_set,train_result)) # 进行预测
    conm=confusion_matrix_for_adaboost(class_labels, prediction,sep)  # 打印混淆矩阵
    describe(conm)
    # errArr = np.mat(np.ones((len(class_labels), 1)))
    # errArr[prediction==class_labels]=0
    #accuracyy=1-sum(errArr)/len(errArr)
    #print("accuracy:",accuracyy)

    return train_result,prediction,class_labels,sep

def adaboostTrain(train_data, numIt = 3):#构建adaboost后的分类器
    """
    数据集，类别标签，迭代次数
    """
    weakClassArr = []#在循环之前创建一个大list
    m=getdatalength(train_data)
    D = np.mat(np.ones((m, 1))/m)
    aggClassEst = np.mat(np.zeros((m, 1)))
    errArr = np.mat(np.ones((m, 1)))
    test_data_length = getdatalength(train_data)
    test_set=load_data(train_data, test_data, D)[0]
    distinct_class_labels=separatebyclass1(test_set).keys()
    # dictionary = dict(zip(distinct_class_labels, np.zeros((len(distinct_class_labels), 1))))
    weakClassArr=[dict() for i in range(test_data_length)]
    for i in range(test_data_length):
        weakClassArr[i]=dict(zip(distinct_class_labels, np.zeros((len(distinct_class_labels), 1))))

    for i in range(numIt):
        print("Round:",i+1)
        train_result, prediction, class_labels,sep = buildNaiveBayes(train_data,train_data,D)
        classEst = prediction
        classEst=np.reshape(classEst,(len(prediction),1))#把
        classLabels=class_labels
        classLabels=np.reshape(classLabels,(len(class_labels),1))

        errArr[prediction == class_labels] = 0
        error = D.T * errArr
        # print("D:", D.T)
        alpha = float(0.5*np.log((1.0-error)/max(error, 1e-16)))
        for i in range(len(prediction)):
            weakClassArr[i][prediction[i]] += alpha
        # print ("classEst:",classEst.T)
        expon = np.multiply(-1*alpha*np.mat(classLabels).T, classEst)
        D = np.multiply(D,np.exp(expon))
        D = D[0]/D[0].sum()
        D=D.T
# #         aggClassEst += alpha*classEst
#         print("aggClassEst: ",aggClassEst.T)
#         aggErrors = np.multiply(np.sign(aggClassEst) != np.mat(classLabels).T, np.ones((m, 1)))
#         ErrorRate = aggErrors.sum()/m
#         print ("total error:",ErrorRate,"\n")
#         if ErrorRate ==0.0:
#             break
    return weakClassArr

def predict_by_adaboost(weakClassArr):#通过分类器进行预测
    prediction_of_adaboost=[]
    for i in range(len(weakClassArr)) :
        prediction_of_adaboost.append(max(weakClassArr[i],key=weakClassArr[i].get))
    return prediction_of_adaboost




if __name__ == "__main__":
    if(len(sys.argv)) != 3:
        print ("usage:{0}".format(sys.argv[0]))
        sys.exit(1)
    else:

        train_file_name = sys.argv[1]
        test_file_name = sys.argv[2]

    train_data = train_file_name
    test_data = test_file_name
    m = getdatalength(train_data)
    D = np.mat(np.ones((m, 1)) / m)
    train_result, prediction, class_labels,sep=buildNaiveBayes(train_data, train_data, D)
    train_result, prediction, class_labels,sep=buildNaiveBayes(train_data, test_data, D)

    print("begin boosting:")
    weakClassArr=adaboostTrain(test_data, numIt = 2)
    prediction_of_adaboost=predict_by_adaboost(weakClassArr)
    # errArr = np.mat(np.ones((len(class_labels), 1)))
    # errArr[prediction_of_adaboost==class_labels]=0
    # #accuracy=1-sum(errArr)/len(errArr)
    # #print("accuracy:",accuracy)
    #
    confusion_matrix=confusion_matrix_for_adaboost(class_labels,prediction_of_adaboost,sep)
    describe(confusion_matrix)
    # # print(prediction_of_adaboost)
    #train_set=load_data(train_data,test_data)[0]#导入训练集

    # for i in range(2):

        # buildNaiveBayes(train_data,train_data,D)
#         classEst=prediction
#         classEst=np.reshape(classEst,(m,1))#把
#         classLabels=test_set[:,0]
#         classLabels=np.reshape(classLabels,(m,1))




#         #bestStump, error, classEst = buildStump(dataArr, classLabels, D)
#         errArr = np.mat(np.ones((m, 1)))
#         errArr[prediction == test_set[:,0]] = 0
#         error=1 - sum(errArr) / len(errArr)
#         if D.T * errArr < error:

#             error = D.T * errArr
#             alpha = float(0.5 * np.log((1.0 - error) / max(error, 1e-16)))
#             print("alpha:", alpha)
#             #         bestStump['alphas'] = alpha
#             #         weakClassArr.append(bestStump)
#             print("classEst:", prediction.T)
#             expon = np.multiply(-1 * alpha * np.mat(classLabels), classEst)
#             D = np.multiply(D, np.exp(expon))
#             D = D / D.sum()
#         else:
#             error=error

#         ErrorRate = error
#         #print("D:", D.T)

#         # aggClassEst += alpha*np.reshape(classEst,(m,1))
#         # print("aggClassEst: ",aggClassEst.T)
#         # aggErrors = np.multiply(np.sign(aggClassEst) != np.mat(classLabels).T, np.ones((m, 1)))
#         # ErrorRate = aggErrors.sum()/m
#         print ("total error:",ErrorRate,"\n")