import numpy as np
import jieba

class WordSequence(object):
    PAD_TAG = '<pad>'
    UNK_TAG = '<unk>'
    START_TAG = '<s>'
    END_TAG = '</S>'

    PAD = 0
    UNK = 1
    START = 2
    END = 3

    def __init__(self):
        '''
        初始化单词到序号的字典。
        是否fit等于false
        这里先要预先存储键值
        填充的
        无符号的标签
        开始标签
        结束标签
        '''
        self.dict = {
            WordSequence.PAD_TAG : WordSequence.PAD,
            WordSequence.UNK_TAG : WordSequence.UNK,
            WordSequence.START_TAG : WordSequence.START,
            WordSequence.END_TAG : WordSequence.END
        }

        self.fited = False

    def to_index(self,word):
        assert self.fited,'WordSequence 尚未进行fit操作'

        if word in self.dict:
            return self.dict[word]
        return WordSequence.UNK

    def to_word(self,index):
        assert self.fited, 'WordSequence 尚未进行fit操作'

        for key,value in self.dict.items():
            if index == value:
                return key
        return WordSequence.UNK_TAG

    def size(self):
        assert self.fited, 'WordSequence 尚未进行fit操作'
        return len(self.dict) + 1

    def __len__(self):
        return self.size()

    def fit(self,sentences,min_count=None,max_count=None,max_features=None):
        '''
        主要是传入句子列表，统计单词词频，然后编号，建立单词到编号之间的映射
        :param sentences:
        :param min_count:
        :param max_count:
        :param max_features:
        :return:
        '''
        assert not self.fited,'WordSequence 只能fit一次'
        assert isinstance(sentences,list),"传入的sentences应该是一个list"

        count = {}

        for sentence in sentences:
            if sentence not in count:
                count[sentence] = 0
            count[sentence] = count[sentence] + 1

        if min_count is not None:
            count = {key:value for key, value in count.items() if value>= min_count}

        if max_count is not None:
            count = {k:v for k,v in count.items() if v<= max_count}

        self.dict = {
            WordSequence.PAD_TAG:WordSequence.PAD,
            WordSequence.UNK_TAG:WordSequence.UNK,
            WordSequence.START_TAG:WordSequence.START,
            WordSequence.END_TAG:WordSequence.END
        }

        #这个max_features 就是取统计词频在前max_feaure的单词
        if isinstance(max_features,int):
            count = sorted(list(count.items()),key=lambda x:x[1])#这个排序后，count元组列表中越靠后的元祖统计词频越高
            if max_features is not None and len(count) > max_features:
                count = count[-int(max_features):] #对，这个应该是取倒数多少个对吧。

            for w,_ in count:
                self.dict[w] = len(self.dict) #这个就是计算单词的编号了。

        else:
            # 这个循环的意思就是为每个单词编个号码
            for w in sorted(count.keys()):
                self.dict[w] = len(self.dict)


        self.fited = True

    def transform(self,sentence,max_length = None,add_end=False):
        '''
        函数的功能是将句子转化成为向量
        也就是将单词列表转化成为单词的编号
        :param sentence:
        :param max_length:
        :return:
        '''
        r = None
        try:
            assert self.fited,'WordSequence 尚未进行fit操作'
            if isinstance(sentence,str):
                sentence = list(sentence)
            if add_end:
                sentence.append(WordSequence.END_TAG)

            if max_length is not None:
                r = [self.PAD] * max_length

            else:
                r = [self.PAD] * len(sentence)

            for index, a in enumerate(sentence):
                if max_length is not None and index >=len(r):
                    break
                r[index] = self.to_index(a)
        except Exception as e:
            print(e)
        return np.array(r)

    def inverse_transform(self,indices,ignore_pad = False,ignore_unk = False,
                          ignore_start = False,ignore_end = False):
        '''
        单词编号，将单词编号转化成为单词
        :param indices:
        :param ignore_pad:
        :param ignore_unk:
        :param ignore_start:
        :param ignore_end:
        :return:
        '''

        ret = []
        for i in indices:
            word = self.to_word(i)
            if word == WordSequence.PAD_TAG and  ignore_pad:
                continue

            if word == WordSequence.UNK_TAG and ignore_unk:
                continue

            if word == WordSequence.START_TAG and ignore_start:
                continue

            if word == WordSequence.END_TAG and ignore_end:
                continue

            ret.append(word)

        return ret


def test():
    ws  = WordSequence()
    ws.fit([
        ['你','好','啊'],
        ['你','好','哦'],
    ])

    indice = ws.transform(['我','们'])
    print(indice)
    words = ws.inverse_transform(indice)
    print(words)

if __name__ =='__main__':
    test()









