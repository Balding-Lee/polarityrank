import preproccess as pp
import generate_graph as gg
import generate_matrix as gm
import polarityrank as pr


if __name__ == '__main__':
    text = '阿龙在成都吃的炸鸡不仅香，而且价格便宜'
    nodes = pp.get_init_attr(text)
    dg = gg.create_network(nodes)
    pr_vec, fu_mat, A_mat = gm.create_matrices(dg)
    pr_vec = pr.calc_polarityrank(pr_vec, fu_mat, A_mat)
    pr.calc_so(pr_vec, nodes)