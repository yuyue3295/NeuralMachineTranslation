import codecs
import tqdm
from word_sequence import WordSequence
import pickle
import re
Chinese_corpors_path = './datas/chinese.txt'
corpors_path = 'datas/corpors_zh.pkl'
def generate_Chinese_ws(Chinese_corpors_path):
    '''
    读取chineses_corpors.txt中的内容，生成中文文本对应的Word_Sequence
    :param Chinese_corpors_path:
    :return:
    '''
    with codecs.open(Chinese_corpors_path,'r','utf-8') as f:
        lines = f.readlines()
        bar = tqdm.tqdm(lines)
        words = []
        for line in bar:
            line = line.strip()
            words.extend(line.split(' '))

        ws = WordSequence()
        ws.fit(words,min_count=2)
        pickle.dump(ws,open(corpors_path,'wb'))
        print(len(ws.dict))

def test_corpors_zh(corpors_path):
    '''

    :param corpors_path:
    :return:
    '''
    ws = pickle.load(open(corpors_path,'rb'))
    while True:
        words = input('请输入一些中文句子:')
        if words.lower() =="quit":
            break
        words = list(words)
        words.insert(0,WordSequence.START_TAG)
        words.append(WordSequence.END_TAG)
        encoded_words = ws.transform(list(words))
        print(encoded_words)

        inverse_words = ws.inverse_transform(encoded_words)
        print(''.join(inverse_words))

def tranform(words,ws,add_end=True):
    assert isinstance(words,list),'words 应该为列表'
    assert isinstance(ws,WordSequence),'ws 应该是WordSequence类'
    if add_end:
        words.append(ws.END_TAG)
    encoded_words = ws.transform(words)
    return encoded_words



def generate_train_zh(chinese_corpors_path='./datas/chinese.txt',corpors_zh_path = './datas/corpors_zh.pkl'):
    '''
    生成
    :param chinese_corpors_path:
    :param corpors_zh_path:
    :return:
    '''
    ws = pickle.load(open(corpors_zh_path,'rb'))
    with codecs.open(chinese_corpors_path,'r','utf-8') as f:
        lines = f.readlines()
        bar = tqdm.tqdm(lines)

        with codecs.open('./datas/zh.train','w','utf-8') as f:

            for line in bar:
                line = line.strip()#这个需要去除掉换行符。
                words = line.split(' ')#语料库中的字是通过/进行划分的。
                encoded_words = tranform(words=words,ws=ws,add_end=True)
                s = '' #这个s就是需要写入zh.train中的编码数字
                for i in encoded_words:
                    s = s + ' '+str(i)

                s = s + '\n'
                f.write(s)
        print('write complete')

    print('done')

def test_zh_train(path='./datas/zh.train'):
    ws = pickle.load(open( './datas/corpors_zh.pkl', 'rb'))
    with codecs.open(path,'r','utf-8') as f:
        lines = f.readlines()
        with codecs.open('./datas/test.txt','w','utf-8') as g:
            bar = tqdm.tqdm(lines)
            for line in bar:
                encoded = []
                line = line.strip()
                for item in line.split(' '):
                    encoded.append(int(item))
                words = ws.inverse_transform(encoded)
                s = ''.join(words) + '\n'
                g.write(s)


def generate_english_ws(corpors_en_path = './datas/english.txt'):
    '''
    生成英文的WordSequence
    :param corpors_en_path:
    :return:
    '''
    with codecs.open(corpors_en_path,'r','utf-8') as f:
        lines = f.readlines()
        bar = tqdm.tqdm(lines)
        totoal_words = []
        for line in bar:
            line = line.strip()
            words = line.split(' ')
            totoal_words.extend(words)

        print('done')
        ws = WordSequence()
        ws.fit(totoal_words,min_count=2)
        print('单词的数目：',len(totoal_words))
        print('预料的行数：',len(lines))
        print('wordsequence的长度是：',len(ws.dict))
        with open('datas/corpors_en.pkl','wb') as g:
            pickle.dump(ws,g)
        print('generate done')

def generate_en_train(corpors_en_path='./datas/corpors_en.pkl',generate_path = './datas/en.train',corpors_path = './datas/english.txt'):
    '''
    生成en.train 文件
    :param corpors_en_path:
    :param generate_path:
    :param corpors_path:
    :return:
    '''
    ws = pickle.load(open(corpors_en_path,'rb'))
    assert isinstance(ws,WordSequence),'ws必须是WordSequence类'

    with codecs.open(corpors_path,'r') as f:
        lines = f.readlines()
        bar = tqdm.tqdm(lines)
        with codecs.open(generate_path,'w','utf-8') as g:
            for line in bar:
                line = line.strip()
                words = line.split(' ')
                encoded = ws.transform(words)
                s = ''
                for item in encoded:
                    s = s + ' '+str(item)
                s = s.strip()
                s = s + '\n'
                g.write(s)
            print('generate')


def test_en_train(corpors_en_path = './datas/corpors_en.pkl',test_path='./datas/en_test.txt',en_train_path = './datas/en.train'):
    '''
    只是测试一下
    :param corpors_en_path:
    :param test_path:
    :return:
    '''
    ws = pickle.load(open(corpors_en_path,'rb'))
    with codecs.open(en_train_path,'r','utf-8') as f:
        lines = f.readlines()
        bar = tqdm.tqdm(lines)
        with codecs.open(test_path,'w','utf-8') as g:
            for line in bar:
                line = line.strip()
                encoded = []
                encoded_charaters = line.split(' ')
                for charactor in encoded_charaters:
                    charactor = int(charactor)
                    encoded.append(charactor)

                s = ws.inverse_transform(encoded)
                s = ' '.join(s)
                s = s + '\n'
                g.write(s)
            print('done')




if __name__ == "__main__":
    generate_Chinese_ws(Chinese_corpors_path)
    # test_corpors_zh(corpors_path)
    generate_train_zh()
    # test_zh_train()
    generate_english_ws()
    generate_en_train()
    # test_en_train()