import numpy as np


def create_matrices(dg):
    """
    创建矩阵, 矩阵包括:
        1. pr_vec: 每个节点PR+与PR-构成的向量, 大小为2n×1
        2. adj_mat: 由权重构成的邻接矩阵, 大小为2n×2n
        3. e_vec: e_i^+和e_i^-构成的向量, 大小为2n×1
        4. fu_mat: 将e_vec复制了2n列向量构成的矩阵, 每一列的向量一模一样, 大小为2n×2n
    :param dg: 有向图
    :return pr_vec: 由PR+和PR-构成的向量, 大小为2n×1
    :return fu_mat: 将e_vec复制了2n列向量构成的矩阵, 每一列的向量一模一样, 大小为2n×2n
    :return A_mat: 由4个归一化权重构成的邻接矩阵拼接而成的大邻接矩阵, 用于矩阵运算, 大小为2n×2n
    """
    pr_vec, pr_plus, pr_minus = create_pr_vec(dg)
    e_vec = create_e_vec(pr_plus, pr_minus)
    fu_mat = create_fu_mat(e_vec)
    A_mat = create_adj_mat(dg)

    return pr_vec, fu_mat, A_mat


def create_pr_vec(dg):
    """
    创建PR向量
    :param dg: 有向图
    :return pr_vec: PR+与PR-组合在一起的向量, ndarray
    :return pr_plus: PR+的向量, 用于求0范数, ndarray
    :return pr_minus: PR-的向量, 用于求0范数, ndarray
    """
    pr_plus = []
    pr_minus = []

    nodes = dg.nodes()
    for node in nodes:
        pr_plus.append(dg.nodes[node]['pr_plus'])
        pr_minus.append(dg.nodes[node]['pr_minus'])

    temp_list = pr_plus + pr_minus  # 合并两个列表

    # 将list转为ndarray对象
    pr_plus = np.array(pr_plus)
    pr_minus = np.array(pr_minus)
    pr_vec = np.array(temp_list)

    return pr_vec, pr_plus, pr_minus


def calc_e(sum_pr, pr, n):
    """
    计算e的值

    计算方法：
        1. 获得PR+与PR-的向量
        2. 判断向量的第0范数, 如果0范数为0, 则这n个e为0; 如果0范数不为0, 则归一化不为0的数,
           再乘以n
           归一化方法: e_i = \frac{pr_i}{\sum_{j=1}^n pr_j}
        3. 将所获得的结果组成向量

    :param sum_pr: 传入的PR向量的总和
    :param pr: 传入的PR向量
    :param n: PR值的个数
    :return e: e向量
    """
    if np.linalg.norm(pr, ord=0):
        e = []
        for i in pr:
            e.append((i / sum_pr) * n)
        e = np.array(e)
    else:
        e = np.zeros(n)

    return e


def create_e_vec(pr_plus, pr_minus):
    """
    创建e_i^+和e_i^-构成的向量
    通过调用calc_e(sum_pr, pr)计算

    e的要求: e(e+和e-组合的向量)的第一范数为2n

    :param pr_plus: PR+构成的向量
    :param pr_minus: PR-构成的向量
    :return:
    """
    n = pr_plus.shape[0]  # 获取向量中元素的个数

    sum_pr_plus = sum(pr_plus)
    sum_pr_minus = sum(pr_minus)

    e_plus = calc_e(sum_pr_plus, pr_plus, n)
    e_minus = calc_e(sum_pr_minus, pr_minus, n)

    e_vec = np.concatenate([e_plus, e_minus])

    return e_vec


def get_sum_weights(dg):
    """
    获得邻接矩阵每一行的数据的和并乘2, i.e. 2 * \sum_{i=1}^n\sum_{j=1}^n w_{ij}

    因为原论文中的公式是q_j^+ 加 q_j^-, 和为2
    由于没有A+和A-之分, i.e. 没有q_j^+ 和q_j^- 之分
    但是矩阵是合并了的, 为了保证矩阵第一范数为1, 所以需要乘2
    :param dg: 有向图
    :return sum_weights: dict, {node: sum_weight}, 每一行的总权重,
                         i.e. /sum_{k \in out(v_j)}|p_{jk}|
    """
    sum_weights = {}
    for node_1 in dg.nodes():
        weight = 0
        for node_2 in dg.nodes():
            weight += dg.edges[node_1, node_2]['weight']
        sum_weights[node_1] = weight * 2

    return sum_weights


def create_adj_mat(dg):
    """
    生成邻接矩阵
        邻接矩阵权重计算方式: a_{ij} = w{ij} / w{j}
        w{j}是j列的权重和, 但是由于邻接矩阵是上下对称的, 所以也是第j行的权重和

    该过程实际上是一个归一化的过程
    :param dg: 有向图
    :return A_mat: 2n×2n大小的邻接矩阵, 由4个n×n的邻接矩阵拼接而成
    """
    n = dg.number_of_nodes()
    sum_weights = get_sum_weights(dg)  # 获得每一行的权重总和

    adj_list = []

    for node_1 in dg.nodes():
        sum_weight = sum_weights[node_1]
        for node_2 in dg.nodes():
            a = dg.edges[node_1, node_2]['weight'] / sum_weight
            adj_list.append(a)

    adj_mat = np.array(adj_list).reshape(n, n)

    A_mat = np.vstack((adj_mat, adj_mat))
    A_mat = np.hstack((A_mat, A_mat))

    return A_mat.T


def create_fu_mat(e_vec):
    """
    创建fu^T矩阵
    计算方法:
        f = e / m, m = 2n
        u: 单位向量
    :param e_vec: e+和e-构成的向量, 大小为2n×1
    :return fu^T: 将e_vec复制了2n列向量构成的矩阵, 每一列的向量一模一样, 大小为2n×2n
    """
    m = e_vec.shape[0]

    f = (e_vec / m).reshape((m, 1))
    u = np.ones((1, m))

    return np.dot(f, u)