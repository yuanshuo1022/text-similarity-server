from gensim.models import Word2Vec
import codecs

word_list = [
    ['从', '明', '天', '起', '，', '做', '一', '个', '幸', '福', '的', '人'],
    ['喂', '马', '，', '劈', '柴', '，', '周', '游', '世', '界'],
    ['从', '明', '天', '起', '，', '关', '心', '粮', '食', '和', '蔬', '菜'],
    ['我', '有', '一', '所', '房', '子', '，', '面', '朝', '大', '海', '，', '春', '暖', '花', '开']]

min_count = 1
vector_size = 128  # Corrected from "size" to "vector_size"

model = Word2Vec(word_list, min_count=min_count, vector_size=vector_size)

fw = codecs.open("word_vec.model", "w", "utf-8")
fw.write(str(len(model.wv.index_to_key)) + " " + str(vector_size))  # Corrected this line
fw.write("\n")
for k in model.wv.index_to_key:
    fw.write(k + " " + ' '.join([str(wxs) for wxs in model.wv[k].tolist()]))
    fw.write("\n")
