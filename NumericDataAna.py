import numpy as np
import matplotlib
import matplotlib.pyplot as plt

F1 = open("D:\Data mining\magic04.txt", "r")
List_row = F1.readlines()
list_source = []
for i in range(len(List_row)):
    column_list = List_row[i].strip().split(",")  # 每一行split后是一个列表
    column_list.pop()#去掉最后的g/h
    list_source.append(column_list)  # 加入list_source
b=np.array(list_source)#转化为np数组
b=b.astype(float)#转换为浮点类型
MeanVector=np.mean(b,axis=0)#均值向量
print(MeanVector)
#中心化
center=b-MeanVector
#求内积
innerP=np.dot(center.T,center)
print(innerP/len(center))
#求外积
outP=0
for i in range(len(center)):
    outP = outP+center[i].reshape(len(center[0]),1)*center[i]
print(outP/len(center))
#通过中心化后的向量计算属性1和2的夹角
t=center.T
corr=np.corrcoef(t[0],t[1])
print(corr[0][1])

fig = plt.figure()
ax1 = fig.add_subplot(111)  #设置标题
ax1.set_title('Scatter Plot')
plt.scatter(t[0],t[1])
plt.xlabel('x1')  #设置X轴标签
plt.ylabel('x2') #设置Y轴标签
plt.show()


u=np.mean(b,axis=0)[0]#第一列均值
sig=np.var(b.T[0])#第一列方差
fig = plt.figure()
ax1 = fig.add_subplot(111)  #设置标题
ax1.set_title('ZTFB')
x = np.linspace(u - 3*sig, u + 3*sig, 50)
y_sig = np.exp(-(x - u) ** 2 /(2* sig **2))/(np.sqrt(2*np.pi)*sig)
plt.plot(x, y_sig, "r-", linewidth=2)
plt.show()

#求每一列的方差
list=[]
for i in range(len(b[0])):
    list.append(np.var(b.T[i]))
print(list)
maxIndex=list.index(max(list))
minIndex=list.index(min(list))
print(maxIndex+1)
print(minIndex+1)

#求矩阵两列协方差
pairCov={}
for i in range(9):
    for j in range(i+1,10):
        st=str(i+1)+'-'+str(j+1)
        pairCov[st]= np.cov(b.T[i],b.T[j])[0][1]
print(pairCov)
print(min(pairCov, key=pairCov.get))
print(max(pairCov, key=pairCov.get))
