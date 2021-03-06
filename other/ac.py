# https://blog.csdn.net/zichen_ziqi/article/details/104525900/
import ahocorasick
import time


class AhocorasickNer:
    def __init__(self, user_dict_path):
        self.user_dict_path = user_dict_path
        self.actree = ahocorasick.Automaton()

    def add_keywords(self):
        flag = 0
        with open(self.user_dict_path, "r", encoding="utf-8") as file:
            for line in file:
                word, flag = line.strip(), flag + 1
                self.actree.add_word(word, (flag, word))
        self.actree.make_automaton()

    def get_ner_results(self, sentence):
        ner_results = []
        # i的形式为(index1,(index2,word))
        # index1: 提取后的结果在sentence中的末尾索引
        # index2: 提取后的结果在self.actree中的索引
        for i in self.actree.iter(sentence):
            ner_results.append((i[1], i[0] + 1 - len(i[1][1]), i[0] + 1))
        return ner_results


if __name__ == '__main__':
    ahocorasick_ner = AhocorasickNer(user_dict_path="../data/addr.txt")
    ahocorasick_ner.add_keywords()
    print(ahocorasick_ner.get_ner_results("我今天去深圳市了"))
    # while True:
    #     sentence = input()
    #     ss = time.time()
    #     res = ahocorasick_ner.get_ner_results(sentence)
    #     print("TIME  : {0}ms!".format(round(1000 * (time.time() - ss), 3)))
    #     print("OUTPUT:{0}".format(res))
