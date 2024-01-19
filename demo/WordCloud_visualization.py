#!/usr/bin/python
# -*- coding: UTF-8 -*-
"""
@author:Administrator
@file:WordCloud_visualization.py
@time:2019/07/31
"""

import logging
from gensim import models
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from wordcloud import WordCloud

font_path = 'C:\Windows\Fonts\simkai.ttf'

'''
获取一个圆形的mask
'''
import numpy as np


def get_mask(image_size=(600, 600), coverage=1.1, pixel_density=3):
    # 基于像素密度增加图像大小
    image_size = (image_size[0] * pixel_density, image_size[1] * pixel_density)

    center = (image_size[0] // 2, image_size[1] // 2)

    # 计算覆盖指定数据百分比的半径
    max_radius = min(center[0], center[1])
    radius = int(coverage * max_radius)

    x, y = np.ogrid[:image_size[0], :image_size[1]]
    distance_from_center = (x - center[0]) ** 2 + (y - center[1]) ** 2
    mask = distance_from_center > radius ** 2
    mask = 255 * mask.astype(int)

    return mask


'''
绘制词云
'''
def draw_word_cloud(word_cloud):
    wc = WordCloud(font_path=font_path, background_color="white", mask=get_mask())
    wc.generate_from_frequencies(word_cloud)
    # 隐藏x轴和y轴
    plt.axis("off")
    plt.imshow(wc, interpolation="bilinear")
    plt.show()


# def draw_word_men(word_cloud):
#     # 创建一个自定义形状的蒙版图像
#     mask = np.array(Image.open('D:/code/python/img/kinjaz.png'))
#
#     # 使用WordCloud库生成词云图像
#     wc = WordCloud(font_path=font_path, background_color='white', mask=mask, contour_width=3, contour_color='black')
#     wc.generate_from_frequencies(word_cloud)
#
#     # 显示词云图像
#     plt.figure(figsize=(8, 8))
#     plt.imshow(wc, interpolation='bilinear')
#     plt.axis('off')
#     plt.show()

font_prop = FontProperties(fname=font_path, size=14)
def draw_word_frequency_bar(word_cloud):
    # 排序单词云字典
    sorted_word_cloud = sorted(word_cloud.items(), key=lambda x: x[1], reverse=True)
    # sorted_word_cloud = sorted(word_cloud.items(), key=lambda x: x[1], reverse=True)
    # 提取单词和关联性
    words, frequencies = zip(*sorted_word_cloud)

    # 创建一个水平条形图
    plt.figure(figsize=(10, 6))
    plt.barh(words[:10], frequencies[:10])  # 显示前10个单词和它们的频率
    plt.xlabel('关联性', fontproperties=font_prop)
    plt.ylabel('单词', fontproperties=font_prop)
    plt.title('单词的关联性条形图', fontproperties=font_prop)  # 使用自定义字体属性
    plt.gca().invert_yaxis()  # 使最常见的单词显示在顶部
    # plt.xticks(fontproperties=font_prop)  # 设置刻度标签字体属性
    plt.yticks(fontproperties=font_prop)  # 设置刻度标签字体属性
    plt.show()





def test():
    logging.basicConfig(format="%(asctime)s:%(levelname)s:%(message)s", level=logging.INFO)
    #加载模型
    model = models.Word2Vec.load("../model/one.model")
    # 输入一个词找出相似的前100个词
    one_corpus = ["帅"]
    result = model.wv.most_similar(one_corpus[0], topn=100)
    # 将返回的结果转换为字典,便于绘制词云
    word_cloud = dict()
    for sim in result:
        # print(sim[0],":",sim[1])
        word_cloud[sim[0]] = sim[1]
    # 绘制词云
    # draw_word_men(word_cloud)
    draw_word_cloud(word_cloud)
    draw_word_frequency_bar(word_cloud)
if __name__ == '__main__':
    test()