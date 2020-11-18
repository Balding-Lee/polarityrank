import numpy as np

A_mat = np.array([0, 1 / 2, 0, 1 / 2, 0, 0, 1 / 2, 0, 1 / 2, 0,
                  1, 0, 0, 0, 0, 1, 0, 0, 0, 0,
                  0, 0, 0, 1, 0, 0, 0, 0, 1, 0,
                  1 / 3, 0, 1 / 3, 0, 1 / 3, 1 / 3, 0, 1 / 3, 0, 1 / 3,
                  0, 0, 0, 1, 0, 0, 0, 0, 1, 0,
                  0, 1 / 2, 0, 1 / 2, 0, 0, 1 / 2, 0, 1 / 2, 0,
                  1, 0, 0, 0, 0, 1, 0, 0, 0, 0,
                  0, 0, 0, 1, 0, 0, 0, 0, 1, 0,
                  1 / 3, 0, 1 / 3, 0, 1 / 3, 1 / 3, 0, 1 / 3, 0, 1 / 3,
                  0, 0, 0, 1, 0, 0, 0, 0, 1, 0,
                  ]).reshape((10, 10))

# pr_vec = np.array([0, 0, 1, 0, 1, 0, 0, 0, 0, 0])
#
# e_vec = np.array([0, 0, 5 / 2, 0, 5 / 2, 0, 0, 0, 0, 0]).reshape((10, 1))
#
# u = np.ones((1, 10))
#
# fu_mat = np.dot(e_vec / 10, u)
#
# d = 0.85
#
# for i in range(200):
#     test_left = (1 - d) * np.dot(fu_mat, pr_vec)
#     test_right = d * np.dot(A_mat, pr_vec)
#     pr_new_1 = test_left + test_right
#
#     if d * np.linalg.norm(np.dot(A_mat, pr_new_1 - pr_vec), ord=2) <= d * np.linalg.norm(A_mat, ord=2) * np.linalg.norm(pr_new_1 - pr_vec, ord=2):
#         print("收敛成功!")
#         print(pr_new_1)
#         break
#     print(pr_new_1)
#     pr_vec = pr_new_1
#
#     # B = (1 - d) * fu_mat + d * A_mat
#     # pr_new_2 = np.dot(B, pr_vec)
#
#
#
# print("===============================")
# print(pr_vec)

# A_mat = np.array([0, 1 / 2, 0, 1 / 2, 0,
#                   1, 0, 0, 0, 0,
#                   0, 0, 0, 1, 0,
#                   1 / 3, 0, 1 / 3, 0, 1 / 3,
#                   0, 0, 0, 1, 0]).reshape((5, 5))

print(A_mat)

print(np.linalg.norm(A_mat.T, ord=1))