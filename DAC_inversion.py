# 实现分治算法计算inversion数量

import pandas as pd
import numpy as np
import time
time0 = time.time()
data = pd.read_table('C:/Users/gulia/Desktop/IntegerArray.txt', header=None)

# print(data)

data_list = data[0].tolist()
# data_list = [7,6,5,4,3,2]
# 分成n份
n = 100000
# n = 6
every_team_number = len(data_list)/n
temp = 0
data_dict = {}
for i in range(n):
    print(i)
    data_dict[i] = data_list[int(temp):int(temp+every_team_number)]
    temp = temp + every_team_number

inversion = 0
def merge(data_dict,inversion ):
    # 每两个数组进行一次合并
    def merge_two(data_dict, inversion):
        ii = 0
        dict_merge = {}
        for i, v in data_dict.items():
            # print('已到data_dict', i, v)
            if i % 2 == 0:
                m = 0
                n = 0
                # 顺序查看每组中的数据，依次加进dict_merge,如果第二个数组中的数写进 合并数组，那么第一个数组中还剩几个元素，inversion就加几
                dict_merge[ii] = []
                while (m < len(data_dict[i])) or (n < len(data_dict[i + 1])):
                    # 1,第一个数组先 合并进merge，现在都是合并第二个数组中的数，第一个数组中的数量相当于0
                    if m == len(data_dict[i]) and n < len(data_dict[i + 1]):
                        dict2_c = data_dict[i + 1][n]
                        dict_merge[ii].append(dict2_c)
                        n = n + 1

                    # 2,第二个数组先 合并进merge
                    elif m < len(data_dict[i]) and n == len(data_dict[i + 1]):
                        dict1_c = data_dict[i][m]
                        dict_merge[ii].append(dict1_c)
                        m = m + 1

                    # 3,两个数组都还没合并完
                    else:
                        dict1_c = data_dict[i][m]
                        dict2_c = data_dict[i + 1][n]
                        if dict1_c < dict2_c:
                            dict_merge[ii].append(dict1_c)
                            m = m + 1
                        else:
                            dict_merge[ii].append(dict2_c)
                            n = n + 1
                            inversion = inversion + (len(data_dict[i]) - m)
                ii = ii + 1
                # print('dict_merge', dict_merge)
        return inversion, dict_merge

    dict_len = len(data_dict)
    if len(data_dict) == 1:
        print('result',inversion)
        return inversion, data_dict
    else:
        # 如果含有奇数个数个元素，把最后一个元素合并到倒数第二个元素中
        # 合并的时候也要比较大小
        if dict_len%2 == 1:
            data_dict_e = {}
            data_dict_e[0] = data_dict[dict_len - 2]
            data_dict_e[1] = data_dict[dict_len - 1]
            inversion_end, data_dict_end = merge_two(data_dict_e, 0)
            inversion = inversion + inversion_end
            data_dict[dict_len-2] = data_dict_end[0]
            del data_dict[dict_len-1]
        else:
            pass
        inversion, dict_merge = merge_two(data_dict, inversion)
        data_dict = dict_merge
        merge(data_dict, inversion)

inversion_number = merge(data_dict,inversion)
print(inversion_number)
time1 = time.time()
print(time1 - time0)