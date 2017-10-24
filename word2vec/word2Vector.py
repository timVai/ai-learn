#-*- encoding:utf-8 -*-
import jieba
from gensim.models import word2vec
import numpy as np
import sys
reload(sys)
sys.setdefaultencoding('utf-8')
'''
先切词——词典是NER以后的词，将预处理好的文件进行切词，以空格的形式分开！
'''
#创建用户自定义词典
def userDict():
    with open("dict.txt","w") as f1:
        out=[]
        with open("userdict.txt","r")as f2:
            lines=f2.readlines()
            for line in lines:
                for item in line.strip().split():
                    out.append(item)
        out=list(set(out))#去重
        for it in out:
            f1.write(it+"\n")
# # 创建停用词list
# def stopwordslist(filepath):
#     stopwords = [line.strip() for line in open(filepath, 'r', encoding='utf-8').readlines()]
#     return stopwords
# 对文件的每句话进行切词！
def cutFile():
    jieba.load_userdict("dict.txt")  # 加载用户自定义词典
    with open("fenDone.txt","w") as f3:
        with open("corpus.txt","r")as f2:
            for line in f2.readlines():
                outStr=""
                wordsList=list(jieba.cut(line.strip()))
                for wd in wordsList:
                    outStr +=wd
                    outStr+=" "
                f3.write(outStr+"\n")
'''
切词后，word2Vector！
'''
def word2Vec():
    sentences = word2vec.Text8Corpus("fenDone.txt")#输入是分词后的文件！
    model = word2vec.Word2Vec(sentences,min_count=1,size=50)#训练！确定需要参与训练词的频次最小1——隐层节点50！
    #
    dicts=[]
    with open("userdict.txt")as f1:
        for line in f1.readlines():
            line = line.decode("utf-8")
            for i in line.strip().split():
                dicts.append(i)
    dicts=list(set(dicts))#去重
    print(len(dicts))
    #计算dict里面的每一个词的Vector！
    with open("vectors.txt", "w")as f2:
        for wd in dicts:
            outstr = ""
            outstr += wd
            try:
                vec = model[wd].tolist()
            except Exception, e:
                vec=np.zeros([1,50]).tolist()
            for i in vec:
                outstr+=" "
                outstr+=str(i)
            f2.write(outstr.encode("utf-8") + "\n")
    # for wd in dicts:
    #     print wd
    #     print(model[wd])

if __name__ == "__main__":
    userDict()#生成用户自定义词典
    cutFile()#使用生成的词典进行切词
    word2Vec()#切词后训练一个Word2Vector模型,且对词典中的每一个词生成Vecor
