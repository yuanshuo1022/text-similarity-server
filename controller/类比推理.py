from gensim.models import Word2Vec
import numpy as np

# 加载已训练好的Word2Vec模型
model = Word2Vec.load("D:/code/python/历史模型/2.2G100w/word2vec2p2G.model")

# 定义单词类比函数
def word_analogy(word1, word2, word3, model):
    try:
        # 执行类比推理：word1 - word2 + word3
        result = model.wv.most_similar(positive=[word1, word3], negative=[word2], topn=2)
        return [item[0] for item in result]
    except Exception as e:
        print(f"类比推理失败：{e}")
        return None

# 进行类比推理
word1 = "丈夫"
word2 = "男人"
word3 = "女人"
result = word_analogy(word1, word2, word3, model)

if result:
    print(f"根据{word1} 推理出 {word2} ，那么 {word3} 推理出 {result}")
else:
    print("类比推理失败")

# 示例：找到与"cat"相似的词
similar_words = model.wv.most_similar("猫", topn=5)
print(f"与'cat'相似的词:")
for word, score in similar_words:
    print(f"{word}: {score}")
