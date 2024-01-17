#!/usr/bin/python
# -*- coding: UTF-8 -*-
#维基百科词向量可视化
"""
@author:Administrator
@file:Visualization.py
@time:2019/07/30
"""
import pandas as pd
pd.options.mode.chained_assignment = None
from gensim.models import Word2Vec
from gensim.models.word2vec import LineSentence
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import logging
import numpy as np
from matplotlib.font_manager import FontProperties
font = FontProperties(fname=r"c:\windows\fonts\simsun.ttc", size=14)

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

# wiki_news = open('./data/reduce_zhiwiki.txt', 'r',encoding='utf-8')
# model = Word2Vec(LineSentence(wiki_news), sg=0,size=200, window=10, min_count=500, workers=6)
# print('model训练完成')
model = Word2Vec.load("D:/code/python/历史模型/2.2G100w/word2vec2p2G.model")
import numpy as np  # Import NumPy

def tsne_plot(model, num_words=100):
    "Creates and TSNE model and plots it"
    labels = []
    tokens = []
    # 随机选择num_words个词
    random_words = np.random.choice(model.wv.index_to_key, num_words, replace=False)
    for word in random_words:
        tokens.append(model.wv[word])
        labels.append(word)

    tokens = np.array(tokens)

    tsne_model = TSNE(perplexity=10, n_components=2, init='pca', n_iter=2500, random_state=23)
    new_values = tsne_model.fit_transform(tokens)
    x = []
    y = []
    for value in new_values:
        x.append(value[0])
        y.append(value[1])

    plt.figure(figsize=(16, 16))
    for i in range(len(x)):
        plt.scatter(x[i], y[i])
        plt.annotate(labels[i],
                     fontproperties=font,
                     xy=(x[i], y[i]),
                     xytext=(5, 2),
                     textcoords='offset points',
                     ha='right',
                     va='bottom')
    plt.show()

tsne_plot(model, num_words=3500)

