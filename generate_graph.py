import networkx as nx
import matplotlib.pyplot as plt
import re


plt.rcParams['font.sans-serif'] = ['SimHei']  # 解决plt无法显示中文问题


def get_sogou_news():
    """
    获取搜狗新闻语料

    :return news: 新闻列表
    """
    news = []

    # 该文件是用gbk编码的, 但是文件中一些特殊字符超出了gbk编码范围, 所以采用gb18030,
    # 为了避免无法编码字符, 添加errors='ignore'忽略
    # 解决方案来源: https://blog.csdn.net/lqzdreamer/article/details/76549256
    with open('./data/corpus/news_tensite_xml.smarty.dat', encoding='gb18030', errors='ignore') as f:
        for line in f:
            if re.match('<content>', line):
                line = line.strip().lstrip('<content>').rstrip('</content>')
                if line:
                    news.append(line)

    return news


def create_network(nodes):
    """
    创建强连接图(每条边都是双向连接)
    :param nodes: 节点
    :return dg: 有向图
    """
    dg = nx.DiGraph()

    for i in range(len(nodes)):
        dg.add_nodes_from([(nodes[i], {'pos': i})])  # 创建节点

    # 创建双向边
    for i in nodes:
        for j in nodes:
            if i != j:
                dg.add_edge(i, j)

    # draw_network(dg)
    get_words_sentiment(dg, nodes)
    get_edges_weight(dg)

    return dg


def draw_network(dg):
    """
    可视化有向图

    :param dg: 有向图
    """
    fig, ax = plt.subplots()
    nx.draw(dg, ax=ax, with_labels=True)
    plt.show()


def get_words_sentiment(dg, nodes):
    """
    从清华大学情感词语数据集中获取节点的初始情感极性
    :param dg: 有向图, 用于更新节点的PR值
    :param nodes: 节点
    """
    sentiment_words = []  # 存储情感词典数据的列表
    node_attribute = {}  # 存储节点属性的字典
    with open('./data/sentiment_words/chinese_sentiment_words_with_polarity.txt', 'r') as f:
        # print(f.read())
        for line in f.readlines():
            sentiment_words.append(line.strip('\n').split('\t'))

    # 如果词语在情感词典中, 将情感值赋值给该词语, 否则情感值为0
    for node in nodes:
        for sentiment_word in sentiment_words:
            if node == sentiment_word[0]:
                if float(sentiment_word[1]) > 0:
                    # 词典中情感值 > 0, 则pr+为情感值, pr-为0
                    node_attribute[node] = [float(sentiment_word[1]), 0]
                    break
                elif float(sentiment_word[1]) < 0:
                    # 词典中情感值 > 0, 则pr+为0, pr-为情感值
                    node_attribute[node] = [0, float(sentiment_word[1])]
                    break
            # 词语不在情感词典中, 则pr+与pr-都为0
            node_attribute[node] = [0, 0]

    # 更新有向图节点
    for node in node_attribute.keys():
        dg.add_nodes_from([node], pr_plus=node_attribute[node][0], pr_minus=node_attribute[node][1])


def get_edges_weight(dg):
    """
    获得每条边的权重
    权重计算方法: n(word1, word2) / N
        其中: n(word1, word2)为word1和word2在所有语料中共现的次数; N为语料的条数
    :param dg: 有向图
    :return:
    """
    news = get_sogou_news()
    nodes = list(dg.nodes())

    N = len(news)
    weights = []
    temp_list = []  # 双向边只用计算一次权重

    for node_1 in nodes:
        temp_list.append(node_1)
        # weight = 0
        for node_2 in nodes:
            if node_1 != node_2:
                if node_2 not in temp_list:
                    # 如果节点不同, 且未曾出现, 则计算在语料中产生的频率
                    # 判断是否出现过, 主要是因为双向图, 权重是相同的,
                    # 且如果不处理, 后续会重复遍历语料, 减少运行速度
                    count = 0
                    for new in news:
                        # 判断是否共现
                        if re.search(node_1, new) and re.search(node_2, new):
                            count += 1
                    weight = count / N
                else:
                    continue
            else:
                weight = 0
            weights.append([node_1, node_2, weight])

    # 虽然双向边的权重只用计算一次, 但是在更新图中的权重的时候要更新两条边
    for weight in weights:
        dg.add_edges_from([(weight[0], weight[1], {'weight': weight[2]})])
        dg.add_edges_from([(weight[1], weight[0], {'weight': weight[2]})])
