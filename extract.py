import os

key = [260881, 260622, 1332730]
add = [32786788,
       63833825,
       88613461,
       32911214,
       43333422]
output = []
f = open("./datasets/test/train.txt", "r")
lines = f.readlines()
for each_line in lines:
    li = each_line.split('\t')
    # print(int(li[0]))
    if int(li[0]) in key and int(li[2]) not in add:
        output.append(int(li[2]))
    if int(li[2]) in key and int(li[0]) not in add:
        output.append(int(li[0]))

print(list(set(output)))