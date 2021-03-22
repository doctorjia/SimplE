import os
import pandas as pd
import numpy as np
import random
import shutil

default_repo = './datasets/WN18/'
out_repo = './datasets/WN18_2/'

entity_dict_file = default_repo + 'entity2id.txt'
relation_dict_file = default_repo + 'relation2id.txt'
entity_df = pd.read_csv(entity_dict_file, header=0, sep='\t')
entity_df.columns = ["A","B"]
entity_dict = dict(zip(entity_df["A"], entity_df["B"]))
n_entity = len(entity_dict)
entities = list(entity_dict.values())
print('#entity: {}'.format(n_entity))
print('-----Loading relation dict-----')
relation_df = pd.read_csv(relation_dict_file, header=0, sep='\t')
relation_df.columns = ["A", "B"]
relation_dict = dict(zip(relation_df["A"], relation_df["B"]))
n_relation = len(relation_dict)


training_file = default_repo + 'train.txt'
validation_file = default_repo + 'valid.txt'
test_file = default_repo + 'test.txt'
print('-----Loading training triples-----')
training_df = pd.read_table(training_file, header=None)
training_triples = list(zip([entity_dict[h] for h in training_df[0]],
                            [entity_dict[t] for t in training_df[2]],
                            [relation_dict[r] for r in training_df[1]]))
n_training_triple = len(training_triples)

i = 0

entity_id = 40942
relation_id = 17

shutil.copyfile(default_repo + "train.txt", out_repo + "train.txt")
shutil.copyfile(default_repo + "entity2id.txt", out_repo + "entity2id.txt")

output_file = open(out_repo + "train.txt", "a")
entity_file = open(out_repo + "entity2id.txt", "a")

new_entities = [32786788,
                63833825,
                88613461,
                32911214,
                43333422]

while i < 5:
    generate_entity = new_entities[i]
    entity_id += 1
    entity_dict[generate_entity] = entity_id
    entity_file.write(str(generate_entity) + "\t" + str(entity_id) + "\n")
    i += 1
    a = [260881, 260622, 1332730]
    for b in a:
        relation_temp = list(relation_dict.keys())[random.randint(0, 17):]
        output_file.write(str(generate_entity) + "\t" + str("%08d" % b) + "\t" + relation_temp[0] + "\n")
        training_df.append([generate_entity, b, relation_temp[0]])

output_file.close()
