import tqdm
from neural_machine_translation import make_src_trg_dataset
from neural_machine_translation import NMTModel
import pickle
import tensorflow as tf
import numpy as np
import codecs

en_corpors_path = './datas/en.train'
zh_corpors_path = './datas/zh.train'
batch_size = 100
n_step = int(100000/batch_size) + 1
n_epoch = 100
input_vocab_size = 22811
output_vocab_size = 32689
embedding_size = 300
learning_rate = 0.001
hidden_units= 128
depth = 2
cell_type = 'lstm'
time_major = False
bidirectional = True
use_residual = False
use_dropout = True
optimizer = 'adam'
attention_type = 'Bahdanau'
max_decode_step = 100
parallel_iteration = 80
beam_width = 4

config = tf.ConfigProto(
        allow_soft_placement=True,
        log_device_placement=False
    )

def run_epoch(session,model, losses, train_op, n_step,learning_rate,turn):
    # 训练一个epoch。重复训练步骤直至遍历完Dataset中所有的数据。
    bar = tqdm.tqdm(range(n_step))
    for step in bar:
        try:
            # 运行train_op 并计算损失值。训练数据在main()函数中以Dataset方式提供
            cost, _,lr = session.run([losses, train_op,learning_rate],feed_dict={model.keep_prob_placeholder.name:0.8})

            bar.set_description('epoch {} loss={:.6f},lr={:.6f}'.format(
                turn,
                np.mean(cost),
                lr
            ))
        except tf.errors.OutOfRangeError:
            pass
        finally:
            pass



    model.save(session, save_path='./model/nmt.ckpt', index=turn)

def main():
    model = NMTModel(input_vocab_size=input_vocab_size,
                     target_vocab_size=output_vocab_size,
                     batch_size=batch_size,
                     embedding_size=embedding_size,
                     hidden_units=hidden_units,
                     depth=depth,
                     cell_type=cell_type,
                     use_dropout=use_dropout,
                     use_residual=use_residual,
                     max_decode_step=max_decode_step,
                     parallel_iterations=parallel_iteration,
                     optimizer=optimizer,
                     learning_rate=learning_rate,
                     attention_type=attention_type,
                     bidirectional=bidirectional
                     )
    # 定义输出数据。
    dataset = make_src_trg_dataset(src_path=en_corpors_path, trg_path=zh_corpors_path, batch_size=batch_size)
    iterator = dataset.make_initializable_iterator()
    (encoder_inputs, encoder_inputs_size), (decoder_input, decoder_input_label, decoder_input_size) = iterator.get_next()
    op,loss,lr = model.build_forward(encoder_input=encoder_inputs,encoder_input_size=encoder_inputs_size,decoder_input=decoder_input,
                        decoder_label=decoder_input_label,decoder_input_size=decoder_input_size)
    saver = tf.train.Saver()


    with tf.Session(config=config) as sess:
        init = tf.global_variables_initializer()
        sess.run(init)
        # saver.restore(sess,save_path='./model/nmt.ckpt-1')
        for epoch in range(1,n_epoch+1):
            sess.run(iterator.initializer)
            run_epoch(session=sess,model=model,losses=loss,train_op=op,n_step=n_step,learning_rate=lr,turn = epoch)

def test():
    '''
    加载seq2seq模型，输入编码后的英文，解码生成中文翻译，打印的是原来的英文句子和预测的中文句子。
    :return:
    '''
    model = NMTModel(input_vocab_size=input_vocab_size,
                     target_vocab_size=output_vocab_size,
                     batch_size=batch_size,
                     embedding_size=embedding_size,
                     hidden_units=hidden_units,
                     depth=depth,
                     cell_type=cell_type,
                     use_dropout=use_dropout,
                     use_residual=use_residual,
                     max_decode_step=max_decode_step,
                     parallel_iterations=parallel_iteration,
                     optimizer=optimizer,
                     beam_width=beam_width,
                     learning_rate=learning_rate,
                     attention_type=attention_type,
                     bidirectional=bidirectional
                     )
    # 定义输出数据。
    dataset = make_src_trg_dataset(src_path=en_corpors_path, trg_path=zh_corpors_path, batch_size=batch_size)
    iterator = dataset.make_initializable_iterator()
    (encoder_inputs, encoder_inputs_size), (decoder_input, decoder_input_label, decoder_input_size) = iterator.get_next()#从数据集中获取编码后的英文句子，以及英文句子的长度
    predict,prob,length = model.build_predict(encoder_inputs,encoder_inputs_size) #输入编码后的英文句子和英文句子的长度，获得预测的句子和预测句子的长度
    saver = tf.train.Saver()
    with tf.Session(config=config) as sess:
        zh_ws = pickle.load(open('./datas/corpors_zh.pkl', 'rb'))#加载中文的word2sequence，这里是预测的中文句子返转码成中文。
        en_ws = pickle.load(open('./datas/corpors_en.pkl','rb'))##加载中文的word2sequence，把编码的英文翻转码成英文句子
        sess.run(iterator.initializer)
        saver.restore(sess=sess,save_path='./model/nmt.ckpt-51')
        with codecs.open('./transform_result.txt','w','utf-8') as f:

            while True:
                result,result_length,english_inputs,eng_len = sess.run([predict,length,encoder_inputs,encoder_inputs_size], feed_dict={model.keep_prob_placeholder.name:1})#返回的result 形状是100x4x句子长度 result_length形状是100x4
                index = 0
                # for i in result:
                #     print(' '.join(en_ws.inverse_transform(english_inputs[index][:eng_len[index]])))#根据英文句子的长度，截取编码的英文，翻转码并打印
                #     print(''.join(zh_ws.inverse_transform(i[:result_length[index]])))
                #     print()
                #     index = index + 1

                for index in range(batch_size):
                    decode_zhs = result[index]
                    decode_lens = result_length[index]
                    print(' '.join(en_ws.inverse_transform(english_inputs[index][:eng_len[index]])))  # 根据英文句子的长度，截取编码的英文，翻转码并打印
                    i = 0
                    for zh in decode_zhs: #因为beamsearch的长度为4,所以输入一句英文，会返回4句中文，下面是一次打印
                        print(''.join(zh_ws.inverse_transform(zh[:decode_lens[i]])))
                        i = i + 1
                    print()





                # print(beam_prob)





if __name__ == "__main__":
    # main()
    test()







