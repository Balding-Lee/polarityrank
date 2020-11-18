import numpy as np


def calc_polarityrank(pr_vec, fu_mat, A_mat):
    """
    polarityrank算法
    计算方法:
        pr_vec := B * pr_vec
        B = ((1 - d)fu^T + dA)
        f = e / m, u: 单位向量

    使用论文中的计算方法, 可以减少一次矩阵与矩阵的乘法

    收敛判断:
        1. ||PR_{k+1} - PR_{k}|| <= 1e-10
        2. 迭代200次
    :param pr_vec: PR+和PR-构成的向量, 大小2n×1
    :param fu_mat: 将e_vec复制了2n列向量构成的矩阵, 每一列的向量一模一样, 大小为2n×2n
    :param A_mat: 由4个归一化权重构成的邻接矩阵拼接而成的大邻接矩阵, 用于矩阵运算, 大小为2n×2n
    :return pr_vec: 迭代结束后的每个节点的PR值
    """
    d = 0.85
    epsilon = 1e-5

    for i in range(200):
        pr_new = (1 - d) * np.dot(fu_mat, pr_vec) + d * np.dot(A_mat, pr_vec)

        print(i, np.linalg.norm(pr_new - pr_vec, ord=2))
        if np.linalg.norm(pr_new - pr_vec, ord=2) <= epsilon:
            pr_vec = pr_new
            break

        pr_vec = pr_new

    print("运行结束!")
    print("最终收敛的PR值:", pr_vec)
    return pr_vec


def calc_so(pr_vec, nodes):
    """
    计算每个节点的情感极性
    计算方法:
        SO(n) = (PR+(n) - PR-(n)) / (PR+(n) + PR-(n))
    :param pr_vec: 收敛后的PR值
    :param nodes: 节点
    """
    # so_dict = {}
    n = len(nodes)  # 记录节点数

    i = 0
    for node in nodes:
        pr_plus = pr_vec[i]
        pr_minus = pr_vec[i + n]

        # so_dict[node] = (pr_plus - pr_minus) / (pr_plus + pr_minus)
        print(node, " 的情感值为: ", (pr_plus - pr_minus) / (pr_plus + pr_minus))

        i += 1

