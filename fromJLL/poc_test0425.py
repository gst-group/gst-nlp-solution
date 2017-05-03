# -*- coding: utf-8 -*-

import re    
#import string
import jieba  
from sklearn.cross_validation import train_test_split
jieba.load_userdict("D:\\python_bag\\base_data\\userdict_insurance.txt") 

path = 'D:\\python_bag\\base_data'
stopwords = {}.fromkeys([line.rstrip() for line in open(path+'\\stop_word_my.txt')]) #fromkeys字典                                   
filename_list = []  
all_words = {}   # 全词库 {'key':value }  
#########################  
#    分词，创建词库      #  
#########################
import pandas as pd
import numpy as np
#words = pd.read_csv('D:\\python_bag\\pingan_jinfu\\data\\pingan_train_pos.csv',index_col=False, dtype={'level2': np.int64},)
#words= pd.read_csv('D:\\python_bag\\word2ve\\data\\mysay_train3.csv',index_col=False, dtype={'level2': np.int32})
words= pd.read_csv('D:\\python_bag\\pingan_jinfu\\poc_test\\poc_train_0427.csv',index_col=False)
#words = pd.read_csv(path + '\\mysay_train1.csv',index_col=False, dtype={'level2': np.int64},)
category_list = []  
category_list = words['level']
raw_word_list = words['text'] 

#分词函数1,包含去停用词
def stopWord(words):
    words_list = []
    for contents in words:
        wordsList = []  
        contents = re.sub(r'\s+','',contents) # trans 多空格 to 空格  
        contents = re.sub(r'\n','',contents)  # trans 换行 to 空格  
        contents = re.sub(r'\t','',contents)  # trans Tab to 空格    
        for seg in jieba.cut(contents,cut_all=False):  
            seg = seg.encode('utf8')  
            if seg not in stopwords:           # remove 停用词  
                if seg!=' ':                   # remove 空格  
                    wordsList.append(seg)      # create 文件词列表  
        file_string = ' '.join(wordsList)
        #print  file_string             
        words_list.append(file_string)
    return words_list

#分词函数2，不去停用词
def textClean(words):
    words_list = []
    for contents in words:
        contents = re.sub(r'\s+','',contents) # trans 多空格 to 空格  
        contents = re.sub(r'\n','',contents)  # trans 换行 to 空格  
        contents = re.sub(r'\t','',contents)  # trans Tab to 空格 
        words_list.append(contents)
    return words_list
    
#去停用词时候函数        
#words_cut_list = stopWord(raw_word_list )       
#words_cut_list = textClean(raw_word_list )
words_cut_list =raw_word_list.apply(lambda s: ' '.join(jieba.cut(s)))
    
    
# 创建词向量矩阵，创建tfidf值矩阵
from sklearn.feature_extraction.text import TfidfTransformer  
from sklearn.feature_extraction.text import CountVectorizer  

freWord = CountVectorizer(stop_words='english')#该类会将文本中的词语转换为词频矩阵，矩阵元素a[i][j] 表示j词在i类文本下的词频      
transformer = TfidfTransformer()#该类会统计每个词语的tf-idf权值  
fre_matrix = freWord.fit_transform(words_cut_list)#fit_transform是将文本转为词频矩阵    
tfidf = transformer.fit_transform(fre_matrix)#第一个fit_transform是计算tf-idf

import pandas as pd  
feature_names = freWord.get_feature_names()           # 特征名  #获取词袋模型中的所有词语
freWordVector_df = pd.DataFrame(fre_matrix.toarray()) # 全词库 词频 向量矩阵  
tfidf_df = pd.DataFrame(tfidf.toarray())  # tfidf值矩阵

print tfidf_df.shape


# tf-idf 筛选  
#tfidf_sx_featuresindex = tfidf_df.sum(axis=0).sort_values(ascending=False)[:2000].index  
#print len(tfidf_sx_featuresindex)  #tfidf提取
tfidf_sx_features_index = freWordVector_df.sum(axis=0).sort_values(ascending=False)[:3500].index 

freWord_tfsx_df = freWordVector_df.ix[:,tfidf_sx_features_index] # tfidf法筛选后的词向量矩阵  
#print freWord_tfsx_df
df_columns = pd.Series(feature_names)[tfidf_sx_features_index] #分词列表头
#print  df_columns
print df_columns.shape  

tfidf_df_1 = freWord_tfsx_df
tfidf_df_1.columns = df_columns  
from sklearn import preprocessing  
le = preprocessing.LabelEncoder()#标签编码（Label encoding）
#tfidf_df_1['label'] = le.fit_transform(category_list)
#label = le.fit_transform(category_list)
label = category_list


#加载数据集，切分数据集80%训练，20%测试  
x_train, x_test, y_train, y_test = train_test_split(tfidf_df_1,label, test_size = 0.2)   
index_test = x_test.index
text_test = words['text'][index_test]
#sizes = 5800
#x_train, x_test =tfidf_df_1[:sizes],tfidf_df_1[(sizes):-1]
#y_train, y_test =label[:sizes],label[(sizes):-1] 

'''
#随机森林
from sklearn.ensemble import RandomForestClassifier
alg = RandomForestClassifier(min_samples_leaf=3, n_estimators=251, random_state=80)
alg.fit(x_train,y_train)
predict = alg.predict(x_test)
pre_forest = alg.predict(tfidf_df_1)

probs = alg.predict_proba(x_test)
predict_probs= probs.max(axis=1) #挑选概率最大的一列
#results=np.c_[y_test,predict]
print((y_test== predict).mean())
'''

#调用MultinomialNB分类器
from sklearn.naive_bayes import MultinomialNB    
clf = MultinomialNB().fit(x_train, y_train)  
doc_class_predicted = clf.predict(x_test)  
pre_NB = clf.predict(tfidf_df_1)
  
print(np.mean(doc_class_predicted == y_test)) 


# 逻辑回归
from sklearn.linear_model import LogisticRegression as LR
clf = LR(random_state=123, class_weight='balanced')
clf = clf.fit(x_train, y_train)
doc_class_predict_LR = clf.predict(x_test) 
pre_lr = clf.predict(tfidf_df_1)
probs_lr = clf.predict_proba(tfidf_df_1).max(axis=1)
  
print(np.mean(doc_class_predict_LR == y_test))

sum_pre = words[['num','text','level']]
#sum_pre['pre_forest'] = pre_forest
sum_pre['pre_NB'] = pre_NB
sum_pre['pre_lr'] = pre_lr
sum_pre['probs_lr'] = probs_lr
#print(np.mean(pre_forest == words['level']))
print(np.mean(pre_NB == words['level']))
print(np.mean(pre_lr == words['level']))

#sum_pre.to_csv('D:\\python_bag\\pingan_jinfu\\poc_test\\pre_word_sum0425_1.csv',encoding='utf-8')

#测试汇总
pre_word_test =pd.DataFrame({'test_text':text_test,'level':y_test})
#pre_word_test['predict_FR']=predict
pre_word_test['predict_NB']=doc_class_predicted
pre_word_test['predict_FR']=doc_class_predict_LR

#pre_word_test.to_csv('D:\\python_bag\\pingan_jinfu\\poc_test\\pre_word_test0425_1.csv',encoding='utf-8')

#筛选特征
"""
rlr = RLR() #建立随机逻辑回归模型，筛选变量  
rlr.fit(x, y) #训练模型  
rlr.get_support() #获取特征筛选结果，也可以通过.scores_方法获取各个特征的分数  
print(u'通过随机逻辑回归模型筛选特征结束。')  
print(u'有效特征为：%s' % ','.join(data.columns[rlr.get_support()]))  
x = data[data.columns[rlr.get_support()]].as_matrix() #筛选好特征  
"""

              
"""
for  i in range(3,12,2):
    for j in range(30,120,10):
        alg = RandomForestClassifier(min_samples_leaf=i, n_estimators=251, random_state=j)
        alg.fit(x_train,y_train)
        predict = alg.predict(x_test)
        results=np.c_[y_test,predict]
        print(i, j,(y_test == predict).mean()) 
"""