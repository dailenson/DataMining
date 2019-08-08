import numpy as np
from sklearn.tree import DecisionTreeClassifier

F1 = open("D:\Data mining\iris.txt", "r")  # 转化为numpy数组
List_row = F1.readlines()
list_source = []
LabelList = []
for i in range(len(List_row)):
    column_list = List_row[i].strip().split(",")  # 每一行split后是一个列表
    LabelList.append(column_list.pop())  # 得到最后的分类属性
    list_source.append(column_list)  # 加入list_source
x = np.array(list_source)  # 转化为np数组
x = x.astype(float)  # 转换为浮点类型
y = np.array(LabelList) #label标签
estimator = DecisionTreeClassifier(criterion='entropy', splitter='best', min_impurity_split=0.05, max_leaf_nodes=5)
estimator = estimator.fit(x, y)

n_nodes = estimator.tree_.node_count #节点数量
children_left = estimator.tree_.children_left #树的左节点
children_right = estimator.tree_.children_right #树的右节点
node_depth = np.zeros(shape=n_nodes, dtype=np.int64) #节点深度
is_leaves = np.zeros(shape=n_nodes, dtype=bool)   #判断是否为叶节点
Entropy=estimator.tree_.impurity
n_size=estimator.tree_.n_node_samples  #每个节点大小
values=estimator.tree_.value   #节点各个类别的数值
threshold = estimator.tree_.threshold #决策值
feature = estimator.tree_.feature
class_names=["Iris-setosa","Iris-versicolor","Iris-virginica"]
feature_names=['x1','x2','x3','x4']#四列属性


stack = [(0, -1)]  # seed is the root node id and its parent depth
while len(stack) > 0:
    node_id, parent_depth = stack.pop()
    node_depth[node_id] = parent_depth + 1
    if (children_left[node_id] != children_right[node_id]):
        stack.append((children_left[node_id], parent_depth + 1))
        stack.append((children_right[node_id], parent_depth + 1))
    else:
        is_leaves[node_id] = True

print("The binary tree structure has %s nodes 按层次遍历输出："
      % n_nodes)
for i in range(n_nodes):#遍历整棵树输出所需信息
    if is_leaves[i]:
        labelList = list(values[i][0])
        majorlable=class_names[labelList.index(max(labelList))]#majority label
        purity=max(labelList)/n_size[i]#纯度
        size=n_size[i] #节点大小
        print("%sLeaf: majority label=%s purity=%s size=%s." % (node_depth[i] * "\t",majorlable,purity,size))
    else:
        IndL=children_left[i] #左子树序号
        IndR=children_right[i]#右字子树序号
        print("%sDecision: %s <= %s ,Gain=%s."
              % (node_depth[i] * "\t",
                 feature_names[feature[i]],
                 threshold[i],
                 Entropy[i] - (Entropy[IndL] * n_size[IndL] + Entropy[IndR] * n_size[IndR]) / n_size[i]
                 ))
