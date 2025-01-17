# -*- encoding: utf-8 -*-
"""
@File    :   step3_build_graph.py    
@Contact :   zhujinchong@foxmail.com
@Author  :   zhujinchong
@Modify Time      @Version    @Desciption
------------      --------    -----------
2025/1/17 10:48    1.0         None
"""

import matplotlib
import matplotlib.pyplot as plt
import networkx as nx

from step2_remove_duplicate import load_all_data

# 设置matplotlib正常显示中文和负号
matplotlib.rcParams['font.sans-serif'] = ['SimHei']  # 用黑体显示中文
matplotlib.rcParams['axes.unicode_minus'] = False  # 正常显示负号


def create_graph(idioms: set):
    G = nx.DiGraph()  # 创建有向图
    for idiom in idioms:
        G.add_node(idiom)  # 添加节点
    for i in range(len(idioms)):
        for j in range(len(idioms)):
            if i != j and idioms[i][-1] == idioms[j][0]:  # 判断是否首尾相连
                G.add_edge(idioms[i], idioms[j])  # 添加边
    # 保存：使用 GraphML 格式保存
    nx.write_graphml(G, 'graph.graphml')
    return G


def load_graph(file_path='graph.graphml'):
    G = nx.read_graphml(file_path)
    return G


def visualize_graph(graph):
    # matplotlib 报错 AttributeError: 'FigureCanvasInterAgg' object has no attribute 'tostring_rgb'. Did you mean: 'tostring_argb'?
    # 解决：降低版本 pip install matplotlib==3.9
    plt.figure(figsize=(10, 8))  # 设置画布大小
    # 简单版
    # nx.draw(graph, with_labels=True, node_size=1000)
    # 优化版
    pos = nx.spring_layout(graph)
    nx.draw(graph, pos, with_labels=True, node_size=3000, node_color='skyblue', font_size=10, font_weight='bold', edge_color='gray')
    plt.show()


def find_cycles_from_node(G, start_node):
    visited = set()  # 记录访问过的节点
    path = []  # 记录当前路径
    result_cycles = []  # 返回数据，记录是环的路径
    result_acyclic = []  # 返回数据，记录不是环的路径

    def dfs(node):
        if len(visited) >= 3:
            return
        visited.add(node)
        path.append(node)
        if len(G.out_edges(node)) <= 0:  # 停止
            acyclic = path[1:] if path else []
            if acyclic and acyclic not in result_acyclic:
                result_acyclic.append(acyclic)
        else:
            # G.neighbors(node)表示：从 node 出发可以直接到达的所有节点
            for neighbor in G.neighbors(node):
                if neighbor not in visited:
                    dfs(neighbor)
                elif neighbor in path:  # 发现环
                    cycle = path[1:]
                    if cycle not in result_cycles:
                        result_cycles.append(cycle)
        path.pop()
        visited.remove(node)

    dfs(start_node)
    # print(result_cycles)
    # print(result_acyclic)
    return result_cycles, result_acyclic


def best_match(cycle_nodes, acyclic_nodes):
    """
    最佳方案：
    A是cycle_nodes，有环
    B是acyclic_nodes，无环

    1. A有数据，且存在奇数个，最终能赢；
    2. B有数据，任意；
    3. B无数据，A有数据，且都是偶数，最终要输；
    4. B无数据，A无数据，输了。。
    """
    # 1
    if acyclic_nodes:
        for nodes in acyclic_nodes:
            if len(nodes) % 2 != 0:
                return nodes[0]
    # 2
    if cycle_nodes:
        return acyclic_nodes[0][0]
    # 3
    if acyclic_nodes:
        return acyclic_nodes[0][0]

    # 4
    return "没有成语"


def main_build_graph():
    # 成语列表
    idioms = load_all_data()
    # 创建图
    create_graph(idioms)


def main_search():
    graph = load_graph()
    while True:
        x = input("input: ")
        x = x.strip()
        cycle_nodes, acyclic_nodes = find_cycles_from_node(graph, x)
        # print(cycle_nodes)
        # print(acyclic_nodes)
        res = best_match(cycle_nodes, acyclic_nodes)
        print(res)


if __name__ == '__main__':
    # 成语列表
    # idioms = ['海阔天空', '空穴来风', '风雨同舟', '舟车劳顿', '顿开茅塞', '塞翁失马', '马到成功', '马上来舟']
    # 创建图
    # idiom_graph = create_graph(idioms)
    # 可视化
    # visualize_graph(idiom_graph)
    # 加载图
    # graph = load_graph()
    # 查询
    # start_node = "风雨同舟"
    # cycle_nodes, acyclic_nodes = find_cycles_from_node(graph, start_node)

    # 最后
    # main_build_graph()
    # print(acyclic_nodes)
    main_search()
