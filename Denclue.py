import collections  as coll
from sklearn.base import BaseEstimator, ClusterMixin
import networkx as nx
import numpy as np


def _findattractor(x_t, X, W=None, h=0.1, eps=0.0001):  # 默认每个样本的每个属性权值相等
    error = 99.
    prob = 0.
    x_l1 = np.copy(x_t)
    radius_new = 0.
    radius_old = 0.
    radius_twiceold = 0.  # 取最后三次爬坡的距离差累计和作为吸引子的半径，这个半径内的样本点由于距离吸引子较近认为密度值够大
    iters = 0.
    while True:
        radius_thriceold = radius_twiceold
        radius_twiceold = radius_old
        radius_old = radius_new
        x_l0 = np.copy(x_l1)
        x_l1, density = _step(x_l0, X, W=W, h=h)
        error = density - prob
        prob = density
        radius_new = np.linalg.norm(x_l1 - x_l0)
        radius = radius_thriceold + radius_twiceold + radius_old + radius_new
        iters += 1
        if iters > 3 and error < eps:
            break
    return [x_l1, prob, radius]  # 返回吸引子、吸引子的密度、吸引子半径


def _step(x_l0, X, W=None, h=0.1):
    n = X.shape[0]
    d = X.shape[1]
    superweight = 0.
    x_l1 = np.zeros((1, d))
    if W is None:
        W = np.ones((n, 1))
    else:
        W = W
    for j in range(n):
        kernel = kernelize(x_l0, X[j], h, d)
        kernel = kernel * W[j]
        superweight = superweight + kernel
        x_l1 = x_l1 + (kernel * X[j])
    x_l1 = x_l1 / superweight  # 计算出吸引子
    density = superweight /((h ** d)*np.sum(W))  # 计算出吸引子的密度
    return [x_l1, density]  # 返回吸引子以及吸引子的密度


def kernelize(x, y, h, degree):  # 计算高斯核
    kernel = np.exp(-(np.linalg.norm(x - y) / h) ** 2. / 2.) / ((2. * np.pi) ** (degree / 2))
    return kernel


class DENCLUE(BaseEstimator, ClusterMixin):
    def __init__(self, h=None, eps=1e-4, min_density=0.15):
        self.h = h
        self.eps = eps
        self.min_density = min_density

    def fit(self, X, sample_weight=None):
        if not self.eps > 0.0:
            raise ValueError("eps must be positive.")
        self.n_samples = X.shape[0]
        self.n_features = X.shape[1]
        density_attractors = np.zeros((self.n_samples, self.n_features))
        radii = np.zeros((self.n_samples, 1))
        density = np.zeros((self.n_samples, 1))

        # 初始化窗口h和样本权重
        if self.h is None:
            self.h = np.std(X) / 5
        if sample_weight is None:
            sample_weight = np.ones((self.n_samples, 1))
        else:
            sample_weight = sample_weight

        self.n_attractor = 0

        # 计算大于min_density的吸引子
        for i in range(self.n_samples):
            density_attractors[i], density[i], radii[i] = _findattractor(X[i], X, W=sample_weight,
                                                                      h=self.h, eps=self.eps)
            if (density[i] >= self.min_density):
                self.n_attractor = self.n_attractor + 1
        print("满足阈值的吸引子个数:"+str(self.n_attractor))
        mapp = [0]*self.n_attractor  # 建立原有吸引子到大于min_density的吸引子的映射
        n_a = 0
        for i in range(self.n_samples):
            if (density[i] >= self.min_density):
                mapp[n_a] = i
                n_a = n_a + 1

        # 创建字典用来保存簇的信息
        cluster_info = {}

        # 初始化簇的集合
        g_clusters = nx.Graph()
        for j1 in range(self.n_attractor):
            z = mapp[j1]
            g_clusters.add_node(z, attr_dict={'attractor': density_attractors[z], 'radius': radii[z],
                                               'density': density[z]})

        # 将密度可达的吸引子进行合并
        for j1 in range(self.n_attractor):
            for j2 in (x for x in range(self.n_attractor) if x != j1):
                x = mapp[j1]
                y = mapp[j2]
                if g_clusters.has_edge(x, y):
                    continue
                diff = np.linalg.norm(
                    g_clusters.node[x].get('attr_dict')['attractor'] - g_clusters.node[y].get('attr_dict')[
                        'attractor'])
                if diff <= (g_clusters.node[x].get('attr_dict')['radius'] + g_clusters.node[y].get('attr_dict')['radius']):
                    g_clusters.add_edge(x, y)
        # 相连的部分构成一个簇
        clusters = list(nx.connected_component_subgraphs(g_clusters))  # 获取连通图
        num_clusters = 0

        # 循环所有的连通图
        for clust in clusters:
            # 统计簇的信息
            c_size = len(clust.nodes())
            # 构造该簇的信息存入字典
            cluster_info[num_clusters] = {'sample': clust.nodes(),  # 构成簇的样本节点
                                          'size': c_size,  # 簇的大小
                                          'attrator':density_attractors[clust.nodes()],
                                          }
            num_clusters += 1
        self.clust_info_ = cluster_info
        self.clusters=clusters

        return self
def main():
    F1 = open("D:\Data mining\iris.txt", "r") #转化为numpy数组
    List_row = F1.readlines()
    list_source = []
    CatList=[]
    for i in range(len(List_row)):
        column_list = List_row[i].strip().split(",")  # 每一行split后是一个列表
        CatList.append(column_list.pop())#得到最后的分类属性
        list_source.append(column_list)  # 加入list_source
    b=np.array(list_source)#转化为np数组
    b=b.astype(float)#转换为浮点类型
    #根据分类标签统计分类结果
    CatResult=coll.Counter(CatList)
    print(CatResult)
    #调用denclue计算聚类结果
    ex=DENCLUE(h=np.std(b) / 5, eps=0.0000001, min_density=0.05)
    results=ex.fit(b, sample_weight=None)
    print("簇的个数："+str(len(results.clust_info_)))
    print(results.clust_info_)

    pur=np.array(CatList)
    for clust in results.clusters:#输出每个簇所包含点的ture id
       print(pur[clust.nodes()])


    NodeList=[]#将每个连通图的节点存在list中便于后期计数
    for i in range(len(results.clust_info_)):
        NodeList.append(list(results.clusters[i]))

    TrueList=[]#计算每个簇的纯度
    for i in range(len(NodeList)):
        TrueList.append(pur[NodeList[i]])
        print(max(coll.Counter(TrueList[i]).values())/len(TrueList[i]))

if __name__ == "__main__":
    main()
