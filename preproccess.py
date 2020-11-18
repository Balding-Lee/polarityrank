from nltk import Tree
from stanfordcorenlp import StanfordCoreNLP
from data import POS_explanation as pos_e


def get_init_attr(text):
    """
    获得初始属性, 包括:
        1. 使用stanford coreNLP进行句法树分析, 得到的结果为str
        2. 使用stanford coreNLP进行POS, 得到的结果为list
        3. 使用stanford coreNLP进行NER, 得到的结果为list
    :param text: 需要进行初始处理的文本
    :return: 数据清洗后的结果, i.e. 清洗了命名实体后的所有名词、动词、形容词
    """
    print("开始加载stanford coreNLP")
    nlp = StanfordCoreNLP(r'D:\pycharm\workspace\polarityrank\stanford-corenlp-4.2.0', lang='zh')
    # words_with_tags = nlp.parse(text)  # 句法树分析
    print("开始pos")
    pos = nlp.pos_tag(text)  # POS
    print("pos结束")
    print("开始ner")
    ner = nlp.ner(text)  # NER
    print("ner结束")
    # draw_tree(words_with_tags)
    return clean_data(pos, ner)


def draw_tree(words_with_tags):
    """
    使用stanford coreNLP绘制句法树
    :param words_with_tags: pos后的结果(list)
    """
    Tree.fromstring(words_with_tags).draw()


def clean_data(pos, ner):
    """
    清洗数据, 步骤共2步:
        1. 先对句子进行NER, 清洗掉所有命名实体, 因为命名实体不带有任何情感。
           由于不清楚stanford coreNLP里面的标签, 所以只提取标签为'o'的词语
        2. 提取出除了命名实体之外的所有名词、动词、形容词

    :param pos: pos后的结果(list)
    :param ner: ner后的结果(list)
    :return nodes: (list)数据清洗后的结果, i.e. 清洗了命名实体后的所有名词、动词、形容词
    """
    n_v_adj = pos_e.n_v_adj

    nodes = []  # 存储句法图的词语
    not_ne = []  # 存储非命名实体的词语
    for n in ner:
        if n[1] == 'O':
            not_ne.append(n[0])

    for p in pos:
        if p[0] in not_ne and p[1] in n_v_adj:
            nodes.append(p[0])

    return nodes
