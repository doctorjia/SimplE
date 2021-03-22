import os
import pandas as pd
import numpy as np
import random

catalog3 = "out_test3.txt"
catalog4 = "out_test4.txt"

def bit_product_sum(x, y):
    return sum([item[0] * item[1] for item in zip(x, y)])


def cosine_similarity(x, y, norm=False):
    """ 计算两个向量x和y的余弦相似度 """
    assert len(x) == len(y), "len(x) != len(y)"
    zero_list = [0] * len(x)
    if x == zero_list or y == zero_list:
        return float(1) if x == y else float(0)

    # method 1
    res = np.array([[x[i] * y[i], x[i] * x[i], y[i] * y[i]] for i in range(len(x))])
    cos = sum(res[:, 0]) / (np.sqrt(sum(res[:, 1])) * np.sqrt(sum(res[:, 2])))

    # method 2
    # cos = bit_product_sum(x, y) / (np.sqrt(bit_product_sum(x, x)) * np.sqrt(bit_product_sum(y, y)))

    # method 3
    # dot_product, square_sum_x, square_sum_y = 0, 0, 0
    # for i in range(len(x)):
    #     dot_product += x[i] * y[i]
    #     square_sum_x += x[i] * x[i]
    #     square_sum_y += y[i] * y[i]
    # cos = dot_product / (np.sqrt(square_sum_x) * np.sqrt(square_sum_y))

    return 0.5 * cos + 0.5 if norm else cos  # 归一化到[0, 1]区间内


def read_catalog(catalog):
    entity_info = 0
    relation_info = 0
    cat3 = open(catalog)
    count = 0
    for index, line in enumerate(open(catalog, 'r')):
        count += 1
        if "#entity" in line:
            entity_info = int(line[line.find(':')+2:-1])
        if "#relation" in line:
            relation_info = int(line[line.find(':')+2:-1])

    cat3_matrix = np.zeros((entity_info, 50))
    cat3_relmat = np.zeros((relation_info, 50))

    #print(entity_info,relation_info)

    iter_id = 0
    ent_id = 0
    rel_id = 0

    for ind in cat3:
        if 'entity embedding result:' in ind:
            break
        else:
            iter_id += 1
    #print(iter_id)
    
    flag = False

    while iter_id < count and flag == False and ent_id < entity_info:
        list_cat3 = []
        cnt_tmp = 0
        temp = "&"
        while ']' not in temp:
            cnt_tmp += 1
            cur_line = cat3.readline()
            if 'relation' in cur_line:
                print('ok')
                flag = True
                break
            if ']' not in temp:
                temp = temp + cur_line[:-1]
            else:
                temp = temp + cur_line[:-2]
        temp = temp[1:]
        if len(temp) == 0:
            continue
        if "[" in temp:
            temp = temp[temp.rfind("[") + 1:]
        else:
            temp = temp[1:]
        if "]" in temp:
            temp = temp[:temp.find("]")]
        else:
            temp = temp[:-1]
        list_cat3 = temp.split()
        for it in range(len(list_cat3)):
            cat3_matrix[ent_id][it] = float(list_cat3[it])
        ent_id += 1
        iter_id += cnt_tmp
        #print(list_cat3)
    while iter_id < count and rel_id < relation_info:
        list_cat3 = []
        temp = "&"
        cnt_tmp = 0
        while ']' not in temp:
            cnt_tmp += 1
            cur_line = cat3.readline()
            if ']' not in temp:
                temp = temp + cur_line[:-1]
            else:
                temp = temp + cur_line[:-2]
        temp = temp[1:]
        if "[" in temp:
            temp = temp[temp.rfind("[")+1:]
        else:
            temp = temp[1:]
        if "]" in temp:
            temp = temp[:temp.find("]")]
        else:
            temp = temp[:-1]
        list_cat3 = temp.split()
        #print(temp)

        for it in range(len(list_cat3)):
            cat3_relmat[rel_id][it] = list_cat3[it]

        iter_id += cnt_tmp
        #print(cat3_relmat[rel_id])
        rel_id += 1

    return cat3_matrix, cat3_relmat


catalog3_ent, catalog3_rel = read_catalog(catalog3)
catalog4_ent, catalog4_rel = read_catalog(catalog4)

print(catalog3_ent.shape, catalog3_rel.shape)
print(catalog4_ent.shape, catalog4_rel.shape)

catalog4_ent_t = catalog4_ent[:catalog3_ent.shape[0]]
catalog4_rel_t = catalog3_rel[:catalog3_rel.shape[0]]

catalog4_top3 = catalog4_ent[:3]
catalog3_top3 = catalog3_ent[:3]

catalog4_3to6 = catalog4_ent[3:6]
catalog3_3to6 = catalog3_ent[3:6]

#print(catalog4_top3)

difference = np.sum(np.abs(np.square(catalog4_ent_t-catalog3_ent)/(catalog3_ent.shape[0]*catalog3_ent.shape[1])))
difference2 = np.sum(np.abs(np.square(catalog4_top3-catalog3_top3)))/(catalog3_ent.shape[0]*catalog3_ent.shape[1])
difference3 = []
difference4 = []

for i in range(100):
    catalog4_3to6 = catalog4_ent[3*i:3*i+3]
    catalog3_3to6 = catalog3_ent[3*i:3*i+3]

    difference3.append(np.sum(np.abs(np.square(catalog4_3to6 - catalog3_3to6)))/(catalog3_ent.shape[0]*catalog3_ent.shape[1]))

#for j in range(300):
#    difference4.append(cosine_similarity(catalog4_ent[j:j+1], catalog3_ent[j:j+1]))


print(difference)
print(difference2)
#print(difference3)
print(np.max(difference3), np.min(difference3), np.average(difference3))
#print(np.max(difference4), np.min(difference4), np.average(difference4))