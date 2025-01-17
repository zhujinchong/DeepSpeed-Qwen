# -*- encoding: utf-8 -*-
"""
@File    :   step2_remove_duplicate.py.py
@Contact :   zhujinchong@foxmail.com
@Author  :   zhujinchong
@Modify Time      @Version    @Desciption
------------      --------    -----------
2025/1/17 10:09    1.0         None
"""


def load1(all_data: set):
    data1 = open("data/chinese-idioms-12976.txt", 'r', encoding='utf-8').readlines()
    for x in data1:
        x = x.split(",")[1][1:-1]
        x = x.strip()
        if len(x) == 4:
            all_data.add(x)


def load2(all_data: set):
    data2 = open("data/成语大全(18744).txt", 'r', encoding='utf-8').readlines()
    for x in data2:
        x = x.strip()
        if len(x) == 4:
            all_data.add(x)


def load3(all_data: set):
    data3 = open("data/成语大全（31648个成语解释）.Txt", 'r', encoding='utf-8').readlines()
    for x in data3:
        if x.strip():
            x = x.split("拼音")[0]
            x = x.strip()
            if len(x) == 4:
                print(x)
                all_data.add(x)


def resave_all_data():
    all_data = set()
    load1(all_data)
    load2(all_data)
    load3(all_data)
    with open("data/my_idioms_29086.txt", 'w', encoding='utf-8') as f:
        for x in all_data:
            f.write(x)
            f.write("\n")


def load_all_data(file_path='data/my_idioms_29086.txt') -> list:
    all_data = []
    data = open(file_path, 'r', encoding='utf-8').readlines()
    for x in data:
        if x.strip():
            all_data.append(x.strip())
    return all_data


if __name__ == '__main__':
    # resave_all_data()
    data = load_all_data()
    # print(data)
    print(len(data))

    while True:
        x = input("input: ")
        x = x.strip()
        res = []
        for idiom in data:
            if idiom.startswith(x):
                res.append(idiom)
        print(res[:3])
