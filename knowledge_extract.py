import pandas as pd
import numpy as np
from itertools import combinations
from chemdataextractor.nlp.tokenize import ChemWordTokenizer
import re
from utils import preprocess
from itertools import islice
import jsonlines
from py2neo import Graph, Node, Relationship
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
graph = Graph("bolt://localhost:7687", auth=("neo4j", "1234"))

f = jsonlines.open('data/knowledge/RL/case.jsonl')
dataset = [i['dict'] for i in f]

def create_nodes_and_relationships(data_set):
    # 建立Material，Structure，Method之间的关系
    for data in data_set:
        nodes = {}
        # 对于每个键，创建节点
        for key in data:
            if key != 'Material':
                nodes[key] = [Node(key, name=value) for value in data[key] if data[key]]

        # 创建节点
        for key, node_list in nodes.items():
            for node in node_list:
                graph.merge(node, key, "name")
        # 建立Material，Structure，Method之间的关系
        for material_node in nodes.get("Material", []):
            for material2_node in nodes.get("Material", []):
                if material_node != material2_node:
                    graph.merge(Relationship(material_node, "Material_TO_Material", material2_node))
            for structure_node in nodes.get("Structure", []):
                graph.merge(Relationship(material_node, "Material_TO_Structure", structure_node))
            for method_node in nodes.get("Method", []):
                graph.merge(Relationship(material_node, "Material_TO_Method", method_node))

        # for structure_node in nodes.get("Structure", []):
        #     for method_node in nodes.get("Method", []):
        #         graph.merge(Relationship(structure_node, "RELATED_TO", method_node))

        # 建立Material，Structure，Method与EAB，RL，Thickness之间的关系
        for property_key in ["EAB", "RL", "Thickness"]:
            for node_key in ["Material"]:
                for main_node in nodes.get(node_key, []):
                    for property_node in nodes.get(property_key, []):
                        graph.merge(Relationship(main_node, "HAS_PROPERTY", property_node))

def static(year, res):
    f = jsonlines.open('data/knowledge/RL/'+year+'model.jsonl')
    # 初始化统计字典
    value_counts = {}
    multi_frequency_words = []
    total = 0
    for data in f:
        total += 1
        data = data['dict']
        for key, values in data.items():
            # 如果键是'RL'、'Thickness'或'EAB'，则将值转换为浮点数
            if key in ['RL', 'Thickness', 'EAB']:
                values = [float(value.rstrip('dBmmGHz')) for value in values]

            # 如果键还没有加入到统计字典中，则加入
            if key not in value_counts:
                value_counts[key] = {}

            # 对每个值进行统计
            for value in values:
                if value in value_counts[key]:
                    value_counts[key][value] += 1
                else:
                    value_counts[key][value] = 1
    # 设置阈值
    threshold = 10
    # 输出统计结果
    # for key, counts in value_counts.items():
    datas = value_counts['Structure']
    counts = {key: val for key, val in datas.items() if val >= threshold}
    counts = dict(sorted(counts.items(), key=lambda kv: kv[1], reverse=True)) #排序  数值按照Key(0)值排，非数值按照value(1)值排
    for value, count in counts.items():
        if count >= threshold:
            multi_frequency_words.append(value)
            print(f"{value}({count})")
    # print(multi_frequency_words)

    # 画柱状图  # 过滤掉小于阈值的数据
    # plt.bar(counts.keys(), counts.values())
    top10dict = dict(islice(counts.items(), 20))
    for k, v in top10dict.items():
        if k in res.keys():
            res[k].append(v)

def co_occurrence():
    # 您的多条数据
    datasets = [data['dict'] for data in f]
    # 指定要保留的键
    keys_to_keep = ['Material', 'Structure', 'Method']
    multi_frequency_words = ['porous', 'hierarchical', 'nanosheets', 'nanotubes', 'core shell', 'three dimensional', 'MOFs', 'MOF', 'MXene', 'aerogels', 'microspheres', 'aerogel', 'multifunctional', 'carbonization', 'particles', 'fibers', 'shell', 'metal organic frameworks',
                             'C', 'N', 'Co', 'graphene O', 'Ni', 'paraffin', 'Fe3O4', 'Fe', 'CNTs', 'graphene', 'RGO', 'rGO', 'porous C', 'SiC',
                             'pyrolysis', 'hydrothermal method', 'in situ', 'calcination']
    # 使用字典解析来保留指定键的部分
    datasets = [{key: original_dict[key] for key in keys_to_keep if key in original_dict} for original_dict in datasets]
    # 设置共现阈值
    threshold = 10
    # 提取所有唯一的值并创建索引映射
    # unique_values = set(val for data in datasets for values in data.values() for val in values)
    unique_values = multi_frequency_words
    value_to_index = {val: idx for idx, val in enumerate(unique_values)}
    # 初始化共现矩阵
    co_occurrence_matrix = np.zeros((len(unique_values), len(unique_values)), dtype=int)
    # 统计每个值出现的次数
    value_counts = {val: 0 for val in unique_values}
    for data in datasets:
        for values in data.values():
            for val in values:
                if val in unique_values:
                    value_counts[val] += 1
    # 更新共现矩阵
    for data in datasets:
        for values in data.values():
            # 只考虑出现次数超过阈值的值
            filtered_values = [val for val in values if val in unique_values]
            for val1, val2 in combinations(filtered_values, 2):
                idx1, idx2 = value_to_index[val1], value_to_index[val2]
                co_occurrence_matrix[idx1][idx2] += 1
                co_occurrence_matrix[idx2][idx1] += 1  # 确保矩阵是对称的
    # 打印共现矩阵中大于阈值的共现关系
    co_dict = {}
    for i in range(len(unique_values)):
        for j in range(i+1, len(unique_values)):
            if co_occurrence_matrix[i][j] >= threshold:
                co_dict[list(unique_values)[i]+' and '+list(unique_values)[j]] = co_occurrence_matrix[i][j]
                # print(f"{list(unique_values)[i]} 和 {list(unique_values)[j]} 的共现次数为：{co_occurrence_matrix[i][j]}")
    # 提取键和值
    sorted_dict = dict(sorted(co_dict.items(), key=lambda kv: kv[1]))
    s = pd.Series(sorted_dict)
    # 创建横向柱状图
    s.plot.barh()
    # 设置图形标题和标签
    plt.title('co occurrence of the Material or Structure')
    plt.xlabel('Values')
    plt.ylabel('Keys')
    plt.show()


def show_value():
    # 原始字典列表数据集
    datasets = [data['dict'] for data in f]
    for item in datasets:
        print(item)
        item["RL"] = float(item["RL"][0][:-2]) if item["RL"] else 0
        item["Thickness"] = float(item["Thickness"][0][:-2]) if item["Thickness"] else 0
        item["EAB"] = float(item["EAB"][0][:-3]) if item["EAB"] else 0

    # 转换为NumPy数组
    x, y, z = [], [], []
    for item in datasets:
        i, j, k = item["Thickness"], item["EAB"], item["RL"]
        if i>0 and i<10 and j>0 and j<20 and k>0 and k<100:
            x.append(i)
            y.append(j)
            z.append(k)
    x, y, z = np.array(x), np.array(y), np.array(z)
    # print(np.median(y))
    # print(np.median(z))
    # print(np.mean(y))
    # print(np.mean(z))

    fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    ax = fig.add_subplot(111)
    # 绘制散点图
    ax.scatter(x, z, alpha=1, color='b', s=10)
    # 设置坐标轴标签
    ax.set_xlabel('Thickness', fontsize=11)
    # ax.set_ylabel('EAB', fontsize=11)
    ax.set_ylabel('RL', fontsize=11)

    # 网格线
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.subplots_adjust(left=0.1, right=0.95, top=0.9, bottom=0.1)
    # 显示图形
    plt.show()

def draw():
    years = [2014, 2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023]
    EAB_median = [5.46, 5.92, 5.90, 5.80, 5.73, 5.60, 5.50, 5.44, 5.60, 5.69]
    RL_median = [35.85, 38.70, 38.20, 41.48, 43.55, 44.24, 45.62, 50.42, 50.55, 51.22]
    # EAB_mean = [7.44, 8.13, 7.72, 7.69, 7.58, 7.13, 6.99, 6.69, 6.50, 6.67]
    RL_mean = [36.95,38.19,37.46,39.75,42.67,43.24,45.27,48.66,49.39,50.24]

    # 创建图表
    fig, ax1 = plt.subplots()

    # 绘制第一条曲线
    # ax1.plot(years, EAB_mean, marker='o', color='tab:red', label='EAB')
    # ax1.set_xlabel('years')
    # ax1.set_ylabel('GHz')
    # ax1.tick_params(axis='y', labelcolor='tab:red')

    # 创建第二个 y 轴
    # ax2 = ax1.twinx()

    # 绘制第二条曲线
    ax1.plot(years, RL_mean, marker='o', color='tab:blue', label='RL')
    ax1.set_ylabel('')
    ax1.tick_params(axis='y', labelcolor='tab:blue')

    # ax2.set_ylim([84, 90])
    # ax2.set_ylim([91.8, 93.5])
    # 添加图例
    lines1, labels1 = ax1.get_legend_handles_labels()
    # lines2, labels2 = ax2.get_legend_handles_labels()
    lines = lines1
            # + lines2
    labels = labels1
             # + labels2
    plt.legend(lines, labels)
    # 显示图表
    plt.show()

def database(csv_path, json_path, database_path):
    datas = []
    cwt = ChemWordTokenizer()
    raw = pd.read_excel(csv_path)
    knowledge = jsonlines.open(json_path)
    i = 0
    for dict in knowledge:
        data = {}
        for ab, doi, title, author in zip(raw['Abstract'][i:], raw['DOI'][i:], raw['Article Title'][i:], raw['Authors'][i:]):
            abs = ""
            sentences = re.split(r'(?<!\d)\.(?![a-z])', str(ab))
            '''删掉垃圾句子'''
            for sentence in sentences:
                if len(sentence) > 30 and 'Elsevier' not in sentence and 'All rights reserved' not in sentence and 'open access' not in sentence:
                    abs += sentence + ' .'
            tokens = cwt.tokenize(abs.strip())
            abs = " ".join(tokens)
            abs = preprocess(abs)
            i += 1
            if abs == dict['text']:
                data['DOI'], data['Article Title'], data['Abstract'], data['Authors'], data['knowledge'] = doi, title, abs, author, dict['dict']
                datas.append(data)
                break
            else:
                continue
    with jsonlines.open(database_path, 'w') as writer:
        writer.write_all(datas)

def num():
    f = jsonlines.open('data/knowledge/RL/allmodel.jsonl')
    Materials, Structures, Methods, RLs, Thicknesss, EABs = [], [], [], [], [], []
    datasets = [data['dict'] for data in f]
    for item in datasets:
        Material, Structure, Method, RL, Thickness, EAB = item['Material'], item['Structure'], item['Method'], item['RL'], item['Thickness'], item['EAB']
        for a in Material:
            if a not in Materials:
                Materials.append(a)
        for b in Structure:
            if b not in Structures:
                Structures.append(b)
        for c in Method:
            if c not in Methods:
                Methods.append(c)
        for d in RL:
            if d not in RLs:
                RLs.append(d)
        for e in Thickness:
            if e not in Thicknesss:
                Thicknesss.append(e)
        for f in EAB:
            if f not in EABs:
                EABs.append(f)
    print(len(Materials), len(Structures), len(Methods), len(RLs), len(Thicknesss), len(EABs))
    print(Structures, Methods)



