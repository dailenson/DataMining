import numpy as np
F1 = open("D:\Data mining\iris.txt", "r") #转化为numpy数组
List_row = F1.readlines()
list_source = []
for i in range(len(List_row)):
    column_list = List_row[i].strip().split(",")  # 每一行split后是一个列表
    column_list.pop()#去掉最后的g/h
    list_source.append(column_list)  # 加入list_source
b=np.array(list_source)#转化为np数组
b=b.astype(float)#转换为浮点类型
#计算核矩阵
K=np.zeros((len(b),len(b)))
for h in range(len(b)):
    for j in range(len(b)):
        K[h][j]=np.square(np.dot(b[h],b[j]))
#核矩阵中心化
I=np.eye(len(b),M=None,k=0)#单位矩阵 对角线全为1
center=I-1/len(b)*np.ones((len(b),len(b)))
ceK=np.dot(np.dot(center,K),center)

#核矩阵标准化
w=ceK*I
for i in range(len(b)):
    w[i][i]=1/np.sqrt(w[i][i])
noK=np.dot(np.dot(w,ceK),w)
print(noK)

#将四维向量拓展到十维向量
Fai=np.zeros((len(b),10))
for i in range (len(b)):
    for j in range(4):
        Fai[i][j] = b[i][j]*b[i][j] #前四个属性为平方
    Fai[i][4] = np.sqrt(2) * b[i][0] * b[i][1]
    Fai[i][5] = np.sqrt(2) * b[i][0] * b[i][2]
    Fai[i][6] = np.sqrt(2) * b[i][0] * b[i][3]
    Fai[i][7] = np.sqrt(2) * b[i][1] * b[i][2]
    Fai[i][8] = np.sqrt(2) * b[i][1] * b[i][3]
    Fai[i][9] = np.sqrt(2) * b[i][2] * b[i][3]
print(Fai)
#中心化

faiu=np.mean(Fai,axis=0)#求每一列的均值
Fai=Fai-faiu  #均值向量维度为二 直接相减即可
Ki=np.zeros((len(b),len(b)))

#标准化
length=[]
for i in range (len(b)): #计算每行的模
    length.append(np.sqrt(sum(ei*ei for ei in Fai[i])))
    Fai[i]=Fai[i]/length[i]
#求两两点积
Ki=np.zeros((len(b),len(b)))
for i in range(len(b)):
    for j in range(len(b)):
        Ki[i][j]=np.dot(Fai[i],Fai[j])
print(Ki)

