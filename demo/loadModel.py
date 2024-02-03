from sklearn.metrics.pairwise import cosine_similarity
from nltk import edit_distance
from scipy.spatial import distance
import gensim
from gensim.models.word2vec import PathLineSentences
import numpy as np
import jieba
from datasketch import MinHash, MinHashLSH

# 加载Word2Vec模型
model = gensim.models.Word2Vec.load("../model/one.model")
similar_words = model.wv.most_similar('可爱', topn=20)
for word, score in similar_words:
    print(f'{word}: {score}')
print(model.wv["欢喜"][:3])
# print(model.wv["热爱"])
# 定义函数将文本转化为平均向量
def text_to_average_vector(text, model):
    words = jieba.lcut(text)
    word_vectors = [model.wv[word] for word in words if word in model.wv]

    if word_vectors:
        return np.mean(word_vectors, axis=0)
    else:
        return np.zeros(model.vector_size)  # 返回零向量作为默认值

# 定义函数计算Jaccard相似度
def jaccard_similarity(text1, text2):
    set1 = set(text1)
    set2 = set(text2)
    intersection = len(set1.intersection(set2))
    union = len(set1) + len(set2) - intersection
    return intersection / union

# 定义函数计算相关系数相似度
def correlation_similarity(vec1, vec2):
    corr_coef = np.corrcoef(vec1, vec2)
    return corr_coef[0][1]

# 定义函数计算局部敏感哈希（LSH）相似度
def lsh_similarity(text1, text2, lsh):
    text1_tokens = set(jieba.lcut(text1))
    text2_tokens = set(jieba.lcut(text2))

    minhash1 = MinHash(num_perm=128)
    minhash2 = MinHash(num_perm=128)

    for word in text1_tokens:
        minhash1.update(word.encode('utf-8'))
    for word in text2_tokens:
        minhash2.update(word.encode('utf-8'))

    lsh.insert("text1", minhash1)
    lsh.insert("text2", minhash2)

    result = lsh.query(minhash1)

    return len(result)/len(text1_tokens)

# 选择两个文本示例
text1 = "喜欢"
text2 = "喜欢。"

# 计算平均向量
vec1 = text_to_average_vector(text1, model)
vec2 = text_to_average_vector(text2, model)

# 计算余弦相似度
cosine_sim = cosine_similarity([vec1], [vec2])[0][0]
print(f"余弦相似度：{cosine_sim}")

# 计算编辑距离相似度
edit_dist = edit_distance(text1, text2)
edit_sim = 1 / (1 + edit_dist)
print(f"编辑距离相似度：{edit_sim}")

# 计算Jaccard相似度
jaccard_sim = jaccard_similarity(set(jieba.lcut(text1)), set(jieba.lcut(text2)))
print(f"Jaccard相似度：{jaccard_sim}")

# 计算相关系数相似度
correlation_sim = correlation_similarity(vec1, vec2)
print(f"相关系数相似度：{correlation_sim}")

# 初始化局部敏感哈希（LSH）
lsh = MinHashLSH(threshold=0.5, num_perm=128)

# 计算局部敏感哈希（LSH）相似度
lsh_sim = lsh_similarity(text1, text2, lsh)
print(f"局部敏感哈希（LSH）相似度：{lsh_sim}")

# 计算欧几里德距离相似度
euclidean_dist = distance.euclidean(vec1, vec2)
euclidean_sim = 1 / (1 + euclidean_dist)
print(f"欧几里德距离相似度：{euclidean_sim}")

# 计算平均相似度
average_sim = (euclidean_sim + cosine_sim + edit_sim + jaccard_sim + correlation_sim ) / 5
print(f"平均相似度：{average_sim}")