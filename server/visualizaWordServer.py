import pandas as pd

pd.options.mode.chained_assignment = None
from gensim.models import Word2Vec
from gensim.models.word2vec import LineSentence
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import logging
import numpy as np


class VisualWord:
    # 返回n个3D词向量的坐标
    def visual_3D(self, model, num_words=100):  # 默认100个词向量
        tokens = []
        labels = []
        random_words = np.random.choice(model.wv.index_to_key, num_words, replace=False)
        for word in random_words:
            tokens.append(model.wv[word])
            labels.append(word)

        tokens = np.array(tokens)
        return tokens, labels

    # 词云
    def visual_word_clound(self, word2vec_model, words):
        similar_words = word2vec_model.wv.most_similar(words, topn=20)
        return similar_words

    # 统计图（根据相似度统计）
    def visual_statis(self, model, word):
        result = model.wv.most_similar(word[0], topn=100)
        return result


