import logging
import os.path
import sys
import gensim
import spacy

# 加载 spaCy 中文模型
nlp = spacy.load("zh_core_web_sm")
token_min_len = 3  # 设置 token_min_len
token_max_len = 15  # 设置 token_max_len
lower = True  # 设置 lower

def tokenize(text,token_in_len,token_max_len,lower):
    # 使用 spaCy 进行分词
    doc = nlp(text)
    tokens = [token.text for token in doc]

    # 在这里使用全局变量 token_min_len、token_max_len 和 lower
    # 进行其他必要的处理

    return tokens


if __name__ == '__main__':
    program = os.path.basename(sys.argv[0])
    logger = logging.getLogger(program)

    logging.basicConfig(format='%(asctime)s: %(levelname)s: %(message)s')
    logging.root.setLevel(level=logging.INFO)
    logger.info("运行 %s" % ' '.join(sys.argv))

    if len(sys.argv) < 3:
        print(globals()['__doc__'] % locals())
        sys.exit(1)

    inp, outp = sys.argv[1:3]
    space = " "
    i = 0

    output = open(outp, 'w', encoding='utf-8')


    # 将自定义分词函数传递给 WikiCorpus，不进行词形还原
    wiki = gensim.corpora.WikiCorpus(inp, tokenizer_func=tokenize, dictionary=[])

    for text in wiki.get_texts():
        output.write(space.join(text) + "\n")
        i = i + 1
        if (i % 10000 == 0):
            logger.info("已保存 " + str(i) + " 篇文章。")

    output.close()
    logger.info("处理完毕，共保存 " + str(i) + " 篇文章。")
