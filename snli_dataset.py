import collections
import os
import re
import zipfile


# d2l.DATA_HUB['SNLI'] = ('https://nlp.stanford.edu/projects/snli/snli_1.0.zip',
#     '9fcde07509c7e87ec61c640c1b2753d9041758e4')

# data_dir = d2l.download_extract('SNLI')
def read_snli(data_dir, is_train):
    """读取SNLI数据集"""

    def extract_text(s):
        # 移除括号
        s = re.sub('\\(', '', s)
        s = re.sub('\\)', '', s)
        # 使用一个空格替换两个以上连续空格
        s = re.sub('\\s{2,}', ' ', s)
        return s.strip()

    # 设置标签0：蕴含，1：矛盾，2：无关
    label_set = {

        'entailment': 0, 'contradiction': 1, 'neutral': 2}
    file_name = os.path.join(data_dir, 'snli_1.0_train.txt' if is_train else 'snli_1.0_test.txt')
    with open(file_name, 'r') as f:
        rows = [row.split('\t') for row in f.readlines()[1:]]
    premises = [extract_text(row[1]) for row in rows if row[0] in label_set]
    hypotheses = [extract_text(row[2]) for row in rows if row[0] in label_set]
    labels = [label_set[row[0]] for row in rows if row[0] in label_set]
    return premises, hypotheses, labels


data_dir = "/data/Downloads/data/dataset/snli_1.0/"
train_data = read_snli(data_dir, is_train=True)
for x0, x1, y in zip(train_data[0][:3], train_data[1][:3], train_data[2][:3]):
    print('premise:', x0)
    print('hypothesis:', x1)
    print('label:', y)
