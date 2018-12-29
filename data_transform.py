import codecs
import re
import tqdm
from word_sequence import WordSequence
Chinese_corpus_path = 'datas/chinese.txt'
def transform(Chineses_corpus_path):
    '''
    将分好词的中文预料切分成一个一个的字
    :param Chineses_corpus_path:
    :return:
    '''
    count = 0
    with codecs.open(Chinese_corpus_path,'r','utf-8') as f:
        with codecs.open('./datas/chinese_corpors.txt','w','utf-8') as g:
            lines = f.readlines()
            print(len(lines))
            bar = tqdm.tqdm(lines)
            for line in bar:
                line = line.strip()
                words = []#每一行的字放入到这个列表中
                phrases = line.split(' ')
                for phrase in phrases:
                    if chkwords(phrase):
                        words.extend(list(phrase))
                    else:
                        words.append(phrase)
                s = '/'.join(words)+'\n'
                g.write(s)
    print('done')

def chkwords(phrase):
    '''检查词语是否是英文或者数字，不是纯的英文和数字则返回True，否则则返回False'''
    if not re.match(r'[0-9a-zA-Z.].', phrase):
        return True
    else:
        return False

def test():
    str1 = '1998年/,/经/过/统/一/部/署/,/伊/犁/州/,/地/两/级/党/委/开/始/尝/试/以/宣/讲/团/的/形/式/,/深/入/学/校/,/村/民/院/落/,/田/间/地/头/,/向/各/族/群/众/进/行/面/对/面/宣/讲/.'
    words = str1.split('/')
    print(words)
    ws =WordSequence()
    ws.fit(words)
    print(ws.dict)

if __name__ == '__main__':
    # transform(Chinese_corpus_path)
    # print(chkwords(str('27年')))
    transform(Chinese_corpus_path)
