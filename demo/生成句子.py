# 导入Word2Vec模型
from gensim.models import Word2Vec

# 加载训练好的Word2Vec模型
model = Word2Vec.load("../model/one.model")

# 选择一些种子词
seed_words = ["国王", "男人", "女人", "女王"]

# 为每个种子词找到最相似的词
similar_words = []
for word in seed_words:
    similar = model.wv.most_similar(word, topn=5)
    similar_words.extend([word for word, score in similar])

# 生成句子
generated_sentence = " ".join(similar_words)
print(generated_sentence)