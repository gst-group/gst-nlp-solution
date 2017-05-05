import re
import jieba

#分词函数1,包含去停用词
def stopWord(words, stopwords):
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
    