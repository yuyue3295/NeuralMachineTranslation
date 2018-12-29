import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import pickle
from tensorflow.nn.rnn_cell import GRUCell
from tensorflow.nn.rnn_cell import LSTMCell
from tensorflow.nn.rnn_cell import DropoutWrapper
from tensorflow.nn.rnn_cell import ResidualWrapper
from tensorflow.nn.rnn_cell import MultiRNNCell
from tensorflow.python.client import device_lib
from tensorflow import layers
from tensorflow.contrib import seq2seq
from tensorflow.contrib.seq2seq import AttentionWrapper
from tensorflow.contrib.seq2seq import LuongAttention
from tensorflow.contrib.seq2seq import BahdanauAttention
from tensorflow.contrib.seq2seq import BeamSearchDecoder
from tensorflow.python.ops import array_ops
from word_sequence import WordSequence




MAX_LEN = 69 #限定句子的最大长度是70
SOS_ID = 2 #目标语言词汇表中<sos>id.
END_ID = 3 #结束的标记
batch_size = 100
en_corpors_path = './datas/en.train'
zh_corpors_path = './datas/zh.train'
input_vocab_size = 22811
target_vocab_size = 32689


def make_dataset(file_path):
    '''
    使用Dataset从一个文件中读取一个语言的数据。
    数据的格式为每一行一句话，单词已经转化为单词编号。
    :param file_path:
    :return:
    '''

    dataset = tf.data.TextLineDataset(filenames=file_path)
    ## 根据空格将单词编号切分并放入到一个一维向量。
    dataset = dataset.map(lambda string: tf.string_split([string]).values)
    # tf.string_split函数是根据delimiter将每一行数据切分成。

    #将字符串形式的单词编号转化成为整数,
    dataset = dataset.map(
        lambda string: tf.string_to_number(string, tf.int32))

    #统计每个单词的数量，并与句子内容一起放入到Dataset中。
    dataset = dataset.map(lambda x: (x, tf.size(x)))
    return dataset

def make_src_trg_dataset(src_path, trg_path, batch_size):
    '''
    从源语言文件src_path和目标语言文件trg_path中分别读取数据，并进行填充和
    batching 操作

    '''

    # 首先分别读取源语言数据和目标语言数据。
    src_data = make_dataset(src_path)
    trg_data = make_dataset(trg_path)

    '''
    通过zip操作将两个Dataset和并为一个Dataset。现在将Dataset中每一项数据
    ds，由4个张量组成：
    ds[0][0]是源句子
    ds[0][1]是源句子长度
    ds[1][0]是目标句子
    ds[1][1]是目标句子长度

    '''
    dataset = tf.data.Dataset.zip((src_data, trg_data))

    # 删除内容为空（只包含<eos>）的句子长度和长度过长的句子

    def filter_length(src_tuple, trg_tuple):
        ((src_input, src_len), (trg_label, trg_len)) = (src_tuple, trg_tuple)

        src_len_ok = tf.logical_and(tf.greater(src_len, 1), tf.less_equal(src_len, MAX_LEN))
        trg_len_ok = tf.logical_and(tf.greater(trg_len, 1), tf.less_equal(trg_len, MAX_LEN))

        return tf.logical_and(src_len_ok, trg_len_ok)

    dataset = dataset.filter(filter_length)

    '''
    从上图可知，解码器需要两种格式的目标句子：
        1. 解码器的输入（trg_input）,形式如同"<sos> X Y Z"
        2. 解码器的目标输出（trg_label），形式如同"X Y Z <eos>"
    上面从文件中读到的目标句子是"X Y Z <eos>"的形式，我们需要从中生成"<sos> X Y Z"
    形式并加入到Dataset中。
    '''

    def make_trg_input(src_tuple, trg_tuple):
        ((src_input, src_len), (trg_label, trg_len)) = (src_tuple, trg_tuple)

        trg_input = tf.concat([[SOS_ID], trg_label[:-1]], axis=0)
        return ((src_input, src_len), (trg_input, trg_label, trg_len))

    dataset = dataset.map(make_trg_input)

    # 随机打乱训练数据
    # dataset = dataset.shuffle(10000)

    # 规定填充后输出的数据维度。
    padded_shapes = (
        (tf.TensorShape([None]),  # 源句子长度是未知的向量量
         tf.TensorShape([])),  # 源句子长度是单个数字
        (tf.TensorShape([None]),  # 目标句子（解码器输入）是长度未知的向量
         tf.TensorShape([None]),  # 目标句子（解码器目标输出）是长度未知的向量
         tf.TensorShape([])))  # 目标句子长度是单个数字

    # 调用pad_batch方法进行batching操作。
    batched_dataset = dataset.padded_batch(batch_size, padded_shapes)
    return batched_dataset

VOCAB_SIZE_THRESHOLD_CPU = 50000 #临界值

def _get_available_gpus():
    '''
    获取当前GPU的信息
    :return:
    '''
    local_device_protos = device_lib.list_local_devices() #获取当前设备的信息。
    print("打印GPU信息")
    return [x.name for x in local_device_protos if x.device_type == 'GPU']
def get_embed_device(vocab_size):
    '''根据输入输出的大小，也即是词汇尺寸的临界值来选择阈值CPU，是在CPU上embedding 还是 GPU上进行embedding'''
    gpus = _get_available_gpus()
    if not gpus or vocab_size > VOCAB_SIZE_THRESHOLD_CPU:
        return "/cpu:0"
    return "/gpu:0"


### 定义NMTModel类来描述模型
class NMTModel(object):
    '''
    在模型的初始化函数中定义模型要用到的变量。
    '''
    def __init__(self,
                 input_vocab_size,# 输入词表的大小
                 target_vocab_size,#输出词表的大小
                 batch_size=32, #数据batch的大小
                 embedding_size=300, #输入词表和输出词表embedding的大小维度，这个是word2vector 这个就是嵌入向量的维度
                 hidden_units=256, #这个rnn或者lstm中隐藏层的大小，encoder和decoder是相同的
                 depth=1, #encoder和decoder rnn的层数，
                 beam_width=0, #是beamsearch的超参数，用于解码
                 cell_type='lstm', #rnn的神经元类型，lstm，gru
                 dropout=0.2, #神经元随机失活的概率
                 use_dropout=False,#是否使用dropout
                 use_residual=False,#是否使用residual
                 optimizer='adam',# 使用哪个优化器
                 learning_rate=1e-3,#学习率
                 min_learing_rate=1e-6,#最小的学习率
                 decay_step=50000,#衰减的步数
                 max_gradient_norm=5.0, #梯度正则裁剪的系数
                 max_decode_step=None, #最大的decode长度，可以非常大
                 attention_type='Bahdanau',#使用的attention类型
                 bidirectional = False,#是否是双向的encoder
                 time_major=False,#是否在计算过程中使用时间作为主要的批量数据！！！
                 mode='train',
                 seed=0,#一些层间操作的随机数
                 parallel_iterations=None,# 并行执行rnn循环的次数
                 share_embedding=False,#是否让encoder和decoder共用一个embedding
                 pretrained_embedding=False,#是否需要使用预训练的embedding
                 ):
        self.input_vocab_size = input_vocab_size
        self.target_vocab_size = target_vocab_size
        self.batch_size = batch_size
        self.embedding_size = embedding_size
        self.hidden_units = hidden_units
        self.depth = depth
        self.cell_type = cell_type.lower()
        self.keep_prob = 1.0 - dropout
        self.use_dropout = use_dropout
        self.use_residual = use_residual
        self.attention_type = attention_type
        self.optimizer = optimizer
        self.learning_rate = learning_rate
        self.min_learning_rate = min_learing_rate
        self.decay_step = decay_step
        self.max_gradient_norm = max_gradient_norm
        self.bidirectional = bidirectional
        self.mode = mode
        self.seed = seed
        self.pretrain_embedding = pretrained_embedding

        assert parallel_iterations is not None,'parallel_iteration 需要要传一个整数'
        if isinstance(parallel_iterations,int):
            self.parallel_iterations = parallel_iterations
        else:
            self.parallel_iterations = batch_size
            assert False

        self.time_major = time_major
        self.share_embedding = share_embedding

        #生成均匀分布的随机数，有四个参数，最小值、最大值、随机的种子数（可以为空），类型
        # self.initializer = tf.random_uniform_initializer(
        #     -0.05,0.05,dtype=tf.float32
        # )
        self.initializer = tf.random_normal_initializer()


        assert self.cell_type in ('gru','lstm'),'cell_type 应该是GRU 或者 LSTM'

        if share_embedding:
            assert input_vocab_size == target_vocab_size,'如果share_embedding为True，那么两个vocab_size必须相等'

        #验证操作----------------------------------------------------------------------------
        assert dropout >=0 and dropout < 1,'dropout 的值必须大于等于零小于1'
        assert attention_type.lower() in ('bahdanau','luong'),'attention_type必须是bahdanau或者luong，而不是{}'.format(attention_type.lower())
        assert beam_width < target_vocab_size,'beam_width {}应该小于target_vocab_size {}'.format(beam_width,target_vocab_size)# 这个需要验证参数对不对

        self.keep_prob_placeholder = tf.placeholder(
            tf.float32,
            shape=[],
            name = 'keep_prob'
        )

        self.global_step = tf.Variable(
            0,trainable=False,name='global_step'
        )

        self.use_beamsearch_decode = False
        self.beam_width = beam_width
        self.use_beamsearch_decode = True if self.beam_width >0 else False
        self.max_decode_step = max_decode_step


        assert self.optimizer.lower() in ('adadelta','adam','rmsprop','momentum','sgd'),\
        'optimizer必须是下列之一：adadelta,adam,rmsprop,momentum,sgd'



    def build_single_cell(self,n_hidden,use_residual):
        '''
        构建一个单独的rnn cell
        :param n_hidden: 隐藏层的神经单元数量
        :param use_residual: 是否使用residual wrapper
        :return:
        '''
        if self.cell_type == 'gru':
            cell_type = GRUCell
        else:
            cell_type = LSTMCell
        cell = cell_type(n_hidden)
        #使用self.use_dropout 可以避免过拟合，等等。
        if self.use_dropout:
            cell = DropoutWrapper(
                cell,
                dtype=tf.float32,
                output_keep_prob=self.keep_prob_placeholder,
                seed = self.seed #一些层之间操作的随机数
                )
        #使用ResidualWrapper进行封装可以避免一些梯度消失或者梯度爆炸
        if use_residual:
            cell = ResidualWrapper(cell)
        return cell

    def build_encoder_cell(self):
        '''
            构建单独的编码器cell。
            根据深度，需要多少层网络。
            :return:
        '''

        multi_cell = MultiRNNCell([
            self.build_single_cell(
                self.hidden_units,
                use_residual=self.use_residual
            )
            for _ in range(self.depth)
        ])

        print("in build_encoder_cell")
        print(hasattr(multi_cell, 'output_size'))
        print(hasattr(multi_cell, 'state_size'))
        return multi_cell

    def build_encoder(self):
        '''
        构建编码器
        编码器的cell，初始化embedding_matrix
        :return:
        '''
        with tf.variable_scope('encoder'):
            encoder_cell = self.build_encoder_cell() #首先是创建编码器，根据层数，使用MultiRNNCell

            with tf.device(get_embed_device(self.input_vocab_size)):

            #### ####这里是需要处需要预先加载训练好的embedding的情况##########
                if self.pretrain_embedding:
                    self.encoder_embeddings = tf.Variable(
                        tf.constant(
                            0.0,
                            shape=(self.input_vocab_size,self.embedding_size)
                        ),
                        trainable = True,
                        name = 'embeddings'
                    )#先声明一个encoder_embedding的变量

                    self.encoder_embeddings_placeholder = tf.placeholder(
                        tf.float32,
                        (self.input_vocab_size,self.embedding_size)
                    )#把预先训练好的embedding matrix通过tf.placeholder，运行时通过feed_dict把数据传进来

                    self.encoder_embeddings_init = self.encoder_embeddings.assign(
                        self.encoder_embeddings_placeholder
                    )#通过assign进行赋值
            #### ####这里是需要处需要预先加载训练好的embedding的情况##########

            #### ####不需要加载embedding的情况，这里直接声明embeddings##########
                else:
                    self.encoder_embeddings = tf.get_variable(
                        name = 'embeddings',
                        shape=(self.input_vocab_size,self.embedding_size),
                        initializer=self.initializer,
                        dtype=tf.float32
                    )#通过-0.5到0.5的均匀分布（self.initilaizer）初始化self.encoder_embeddings
            #### ####不需要加载embedding的情况，这里直接声明embeddings##########

            assert self.encoder_inputs is not None,'self.encoder_inputs 不能为空'
            self.encoder_inputs_embedded = tf.nn.embedding_lookup(
                params=self.encoder_embeddings,
                ids=self.encoder_inputs
            )#params传入的是代表embedding matrix的tensor，ids传入的是encoder的输入，编码之后的

            #########这里是使用residual的情况##############################################################
            if self.use_residual:
                self.encoder_inputs_embedded = layers.dense(self.encoder_inputs_embedded,#输入tensor
                                                            self.hidden_units,#Integer or Long,output维度的大小.
                                                            use_bias=False,
                                                            name='encoder_residual_projection')
                #self.encoder_inputs_embedded 是一个全连接层，没有使用激活函数

            #########这里是使用residual的情况##############################################################

            inputs = self.encoder_inputs_embedded ####这里inputs就是embedding vector


            assert inputs is not None,'inputs 不能为空'
            assert self.encoder_inputs_length is not None,'self.encoder_inputs_length不能为空'
            assert isinstance(self.parallel_iterations,int),'self.parallel_iterations 必须为int'

            if self.time_major:
                inputs = tf.transpose(inputs,(1,0,2))

            #########不是双向LSTM解码的情况##############################################################
            if not self.bidirectional:
                (
                    encoder_outputs,
                    encoder_states
                ) = tf.nn.dynamic_rnn(
                    cell = encoder_cell,
                    inputs=inputs,
                    sequence_length=self.encoder_inputs_length,#这个是需要encoder输入编码的长度，sentence的长度。
                    time_major=self.time_major,
                    parallel_iterations=self.parallel_iterations,#并行执行rnn循环的次数
                    swap_memory=True#建议设置为true
                )
                '''
                一般来讲dynamic_rnn它的优点就是，动态rnn的内存可以交换
                '''
            #########不是双向LSTM解码的情况##############################################################

            #########双向LSTM解码的情况##############################################################
            else:
                print('in bidirection')
                encoder_cell_bw = self.build_encoder_cell()
                (
                    (encoder_fw_outputs,encoder_bw_outputs),
                    (encoder_fw_state,encoder_bw_state)
                ) =tf.nn.bidirectional_dynamic_rnn(
                    cell_bw = encoder_cell_bw,#反向LSTM
                    cell_fw = encoder_cell,#前向LSTM
                    inputs=inputs,#embedding vector
                    sequence_length=self.encoder_inputs_length,#输入数据的长度，sentence的长度列表
                    dtype=tf.float32,
                    time_major=self.time_major,
                    parallel_iterations=self.parallel_iterations,#并行执行rnn循环的次数
                    swap_memory=True#建议设置为true
                )


                #需要把双向的RNN的输出结果进行合并
                encoder_outputs = tf.concat(
                    (encoder_fw_outputs,encoder_bw_outputs),2
                )

                encoder_states = []
                for i in range(self.depth):
                    encoder_states.append(encoder_fw_state[i])
                    encoder_states.append(encoder_bw_state[i])
                encoder_states = tuple(encoder_states)

            #########双向LSTM解码的情况##############################################################

        return encoder_outputs,encoder_states

    def build_decoder_cell(self,encoder_outputs,encoder_states):
        '''

        构建解码器的cell,返回一个解码器的cell和解码器初始化状态。
        :param encoder_outputs:

        :param encoder_state:
        :return:
        '''
        encoder_input_length = self.encoder_inputs_length
        batch_size = self.batch_size

        if self.bidirectional:
            encoder_states = encoder_states[-self.depth:]

        if self.time_major:
            encoder_outputs = tf.transpose(encoder_outputs,(1,0,2))

        assert encoder_input_length is not None,'encoder_state_length 不能为空'
        assert isinstance(batch_size,int),'batchsize的值必须为int类型'
        assert encoder_outputs is not None,'encoder_outputs is not None'
        assert encoder_states is not None,'encoder_state is not None'
        #########################使用beamsearch的情况#####################################################
        if self.use_beamsearch_decode:
            '''这个tile_batch 会将tensor复制self.beam_with 份，相当于是
            batch的数据变成了原来的self.beam_width 倍
            '''
            encoder_outputs = seq2seq.tile_batch(
                encoder_outputs,multiplier=self.beam_width
            )
            encoder_states = seq2seq.tile_batch(
                encoder_states,multiplier=self.beam_width
            )
            encoder_input_length = seq2seq.tile_batch(
                self.encoder_inputs_length,multiplier=self.beam_width
            )
            #如果使用了beamsearch，那么输入应该是beam_width的倍数乘以batch_size
            batch_size *=self.beam_width
        #########################使用beamsearch的情况#####################################################


        #########################使用注意力机制###########################################################
        if self.attention_type.lower() == 'luong':
            self.attention_mechanism = LuongAttention(
                num_units=self.hidden_units,
                memory=encoder_outputs,
                memory_sequence_length=encoder_input_length
            )
        else:
            self.attention_mechanism = BahdanauAttention(
                num_units=self.hidden_units,
                memory=encoder_outputs,
                memory_sequence_length=encoder_input_length
            )#双向LSTM的话encoder_outputs 就是它的隐藏状态h1
        #########################使用注意力机制###########################################################

        cell = MultiRNNCell(
            [
                self.build_single_cell(
                    self.hidden_units,
                    use_residual=self.use_residual
                )
                for _ in range(self.depth)
            ])
        #这个cell就是多层的。

        alignment_history = (
            self.mode != 'train' and not self.use_beamsearch_decode
        )
        #alignment_history在不是训练状态以及没有使用beamsearch的时候使用。

        def cell_input_fn(inputs,attention):
            '''
            根据attn_input_feeding属性来判断是否在attention计算前进行一次投影的计算
            使用注意力机制才会进行的运算
            :param inputs:
            :param attention:
            :return:
            '''

            if not self.use_residual:
                print(inputs.get_shape,'inputs_shape')
                print(attention.get_shape,'inputs_shape')
                print(array_ops.concat([inputs,attention],-1),'inputs和attention拼接之后的形状')
                return array_ops.concat([inputs,attention],-1)

            attn_projection = layers.Dense(self.hidden_units,
                                           dtype=tf.float32,
                                           use_bias=False,
                                           name='attention_cell_input_fn')

            '''
            这个attn_projection(array_ops.concat([inputs,attention],-1))我的理解就是
            layers.Dense(self.hidden_units,
                                           dtype=tf.float32,
                                           use_bias=False,
                                           name='attention_cell_input_fn')(array_ops.concat([inputs,attention],-1))
            Dense最终继承了Layer类，Layer中定义了call方法和__call__ 方法，Dense也重写了call方法，__call__方法中调用call方法，call方法中还是起一个全连接层层的作用，__call__
            方法中执行流程是：pre process，call，post process
            '''
            return attn_projection(array_ops.concat([inputs,attention],-1))


        cell = AttentionWrapper(
            cell=cell,
            attention_mechanism=self.attention_mechanism,
            attention_layer_size=self.hidden_units,
            alignment_history=alignment_history,#这个是attention的历史信息
            cell_input_fn=cell_input_fn,#将attention拼接起来和input拼接起来
            name='Attention_Wrapper'
        )#AttentionWrapper 注意力机制的包裹器

        decoder_initial_state = cell.zero_state(
            batch_size,tf.float32
        )#这里初始化decoder_inital_state

        #传递encoder的状态
        decoder_initial_state = decoder_initial_state.clone(
            cell_state = encoder_states
        )

        return cell,decoder_initial_state

    def init_optimizer(self):
        '''
        sgd,adadelta,adam,rmsprop,momentum
        :return:
        '''

        learning_rate = tf.train.polynomial_decay(
            self.learning_rate,
            self.global_step,
            self.decay_step,
            self.min_learning_rate,
            power=0.5
        )
        #初始化学习率的下降算法

        self.current_learning_rate = learning_rate

        #返回需要训练的参数的列表
        trainable_params = tf.trainable_variables()


        #设置优化器
        if self.optimizer.lower() == 'adadelta':
            self.opt = tf.train.AdadeltaOptimizer(learning_rate=learning_rate)

        elif self.optimizer.lower() == 'adam':
            self.opt = tf.train.AdamOptimizer(
                learning_rate = learning_rate
            )
        elif self.optimizer.lower() == 'rmsprop':
            self.opt = tf.train.RMSPropOptimizer(
                learning_rate = learning_rate
            )
        elif self.optimizer.lower() == 'momentum':
            self.opt = tf.train.MomentumOptimizer(
                learning_rate = learning_rate
            )
        elif self.optimizer.lower() == 'sgd':
            self.opt = tf.train.GradientDescentOptimizer(
                learning_rate=learning_rate
            )

        ############################通过self.loss生成的self.updates######################################
        gradients = tf.gradients(self.loss,trainable_params)
        clip_gradients,_ = tf.clip_by_global_norm(
            gradients,self.max_gradient_norm
        )#对梯度下降进行裁剪

        #更新model
        self.updates = self.opt.apply_gradients(zip(clip_gradients,trainable_params),
                                                global_step=self.global_step)
        self.updates1 = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss=self.loss,global_step=self.global_step)

        ############################通过self.loss生成的self.updates######################################

        ############################通过self.loss_rewards生成的self.updates_rewards######################################
        # gradients = tf.gradients(self.loss_rewards,trainable_params)
        # clip_gradients, _ =tf.clip_by_global_norm(
        #     gradients,self.max_gradient_norm
        # )
        # self.updates_rewards = self.opt.apply_gradients(
        #     zip(clip_gradients,trainable_params),
        #     global_step=self.global_step
        # )
        ############################通过self.loss_rewards生成的self.updates_rewards######################################



    def build_decoder(self, encoder_output, encoder_state):
        '''
        构建解码器
        :param encoder_output:
        :param encoder_state:
        :return:
        '''

        with tf.variable_scope('decoder') as decoder_scope:  # 这里是为了调试方便，将参数折叠成一个层。

            #####解码器的单元 解码器的初始化状态########
            print(decoder_scope,'decoder_scope')
            (
                self.decoder_cell,
                self.decoder_initial_state
            ) = self.build_decoder_cell(encoder_output, encoder_state)
            #####解码器的单元 解码器的初始化状态########




            with tf.device(get_embed_device(self.target_vocab_size)):
                ##############################################加载解码器的embedding##################################################
                ###编码器和解码器是否共享embedding matrix###
                if self.share_embedding:
                    self.decoder_embeddings = self.encoder_embeddings
                ###编码器和解码器是否共享embedding matrix###

                ###是否加载预先训练好的embedding matrix###
                elif self.pretrain_embedding:
                    self.decoder_embeddings = tf.Variable(
                        tf.constant(
                            0.0,
                            shape=(self.target_vocab_size,
                                   self.embedding_size)
                        ),
                        trainable=True,
                        name='embeddings'
                    )

                    self.decoder_embeddings_placeholder = tf.placeholder(
                        dtype=tf.float32,
                        shape=(self.target_vocab_size, self.embedding_size),
                    )

                    self.decoder_embeddings_init = self.decoder_embeddings.assign(self.decoder_embeddings_placeholder)
                    #运行时通过placeholder传入embedding matrix，通过assign的形式进行赋值。
                ###是否加载预先训练好的embedding matrix###
                else:
                    self.decoder_embeddings = tf.get_variable(
                        name='embeddings',
                        shape=(self.target_vocab_size, self.embedding_size),
                        initializer=self.initializer,
                        dtype=tf.float32
                    )
                    #不加载预先训练好的embedding matrix的情况，声明一个decoder_embeddings matrix,使用-0.5到0.5的均匀分布进行初始化
                ##############################################加载解码器的embedding##################################################

            # 定义输出的projection，实际上就是全连接层
            ##################################解码器的映射############################################
            self.decoder_output_projection = layers.Dense(
                self.target_vocab_size,
                dtype=tf.float32,
                use_bias=False,
                name='decoder_output_projection'
            )
            ##################################解码器的映射############################################



            #######################解码器的word embedding########################
            self.decoder_inputs_embedded = tf.nn.embedding_lookup(
                params=self.decoder_embeddings,
                ids=self.decoder_inputs_train
            )
            #######################解码器的embedding vector########################

            inputs = self.decoder_inputs_embedded
            if self.time_major:
                inputs = tf.transpose(inputs, (1, 0, 2))

            ############ seq2seq的一个类，用来帮助feeding参数。#######################
            training_helper = seq2seq.TrainingHelper(
                inputs=inputs,#这个是decoder的inputs,不是label
                sequence_length=self.decoder_inputs_length,#用作输入的解码器长度。
                time_major=self.time_major,
                name='training_helper'
            )
            """A helper for use during training.  Only reads inputs.
            在训练期间使用的一个帮助类. 仅仅是读取inputs。
            返回的是sample_ids,RNN 预测输出，通过argmax函数得到的。
              Returned sample_ids are the argmax of the RNN output logits.
          """
            ############ seq2seq的一个类，用来帮助feeding参数。#######################

            training_decoder = seq2seq.BasicDecoder(
                cell=self.decoder_cell,
                helper=training_helper,
                initial_state=self.decoder_initial_state
            )

            ##########################dynamic_decode 真正解码的过程#########################################
            # decoder在当前的batch下的最大time_steps
            max_decoder_length = tf.reduce_max(
                self.decoder_inputs_length
            ) #这个batch中解码器输入的最大长度



            (
                outputs,
                self.final_state,
                final_sequence_lengths
            ) = seq2seq.dynamic_decode(
                decoder=training_decoder,
                output_time_major=self.time_major,
                impute_finished=True,
                maximum_iterations=max_decoder_length,
                parallel_iterations=self.parallel_iterations,
                swap_memory=True,
                scope=decoder_scope
            )

            self.decoder_logits_train = self.decoder_output_projection(
                outputs.rnn_output
            )
            print(outputs.rnn_output.get_shape,'outputs.run_output')
            ##########################dynamic_decode 真正解码的过程#########################################

            '''
            self.masks感觉有用，通过这个mask来区分数据位和填充位，这个是计算sequence_loss需要传入的参数。
          '''
            self.masks = tf.sequence_mask(
                lengths=self.decoder_inputs_length,
                maxlen=max_decoder_length,
                dtype=tf.float32,
                name='masks'
            )

            ############################ 预测值 ##############################
            decoder_logits_train = self.decoder_logits_train
            if self.time_major:
                decoder_logits_train = tf.transpose(decoder_logits_train(1, 0, 2))

            self.decoder_pre_train = tf.argmax(
                decoder_logits_train,
                axis=-1,
                name='deocder_pred_train'
            )


            ############################ 预测值 ##############################


            ##############################损失函数#################################################################

            self.tran_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
                labels=self.decoder_label,#这里应该改成decoder的标签了
                logits=decoder_logits_train
            )

            # self.masks_rewards = self.masks * self.rewards
            #
            # self.loss_rewards = seq2seq.sequence_loss(
            #     logits=decoder_logits_train,
            #     targets=self.decoder_label,#这里应该改成decoder的标签了
            #     weights=self.masks_rewards,
            #     average_across_timesteps=True,
            #     average_across_batch=True
            # )

            self.loss = seq2seq.sequence_loss(
                logits=decoder_logits_train,
                targets=self.decoder_label,#这里应该改成decoder的标签了
                weights=self.masks,# 区分padding位和数据位，这时候需要。
                average_across_timesteps=True,
                average_across_batch=True
            )

            #不再需要使用addloss了，之前使用的

            ##############################损失函数#################################################################





    def build_forward(self,encoder_input, encoder_input_size, decoder_input, decoder_label, decoder_input_size):
        self.encoder_inputs = encoder_input
        self.encoder_inputs_length = encoder_input_size
        self.decoder_inputs_train = decoder_input
        self.decoder_inputs_length = decoder_input_size
        self.decoder_label = decoder_label

        #创建一个decoder，返回encoder_outputs,encoder_final_state
        encoder_outputs, encoder_states = self.build_encoder()
        print(encoder_outputs.get_shape,'encoder_outputs.shape')
        print(encoder_states,'encoder_states')

        #创建一个解码器，可以使用self.loss或者self.loss_reward
        self.build_decoder(encoder_outputs,encoder_states)
        self.init_optimizer()

        self.saver = tf.train.Saver()
        return self.updates,self.loss,self.current_learning_rate

    def run_epoch(self):
        pass

    def predict(self,encoder_output,encoder_state):
        '''
        开始预测
        :param encoder_output:
        :param encoder_state:
        :return:
        '''
        with tf.variable_scope('decoder') as decoder_scope:  # 这里是为了调试方便，将参数折叠成一个层。

            #####解码器的单元 解码器的初始化状态########
            print(decoder_scope,'decoder_scope')
            (
                self.decoder_cell,
                self.decoder_initial_state
            ) = self.build_decoder_cell(encoder_output, encoder_state)
            #####解码器的单元 解码器的初始化状态########


            with tf.device(get_embed_device(self.target_vocab_size)):
                ##############################################加载解码器的embedding##################################################
                ###编码器和解码器是否共享embedding matrix###
                if self.share_embedding:
                    self.decoder_embeddings = self.encoder_embeddings
                ###编码器和解码器是否共享embedding matrix###

                ###是否加载预先训练好的embedding matrix###
                elif self.pretrain_embedding:
                    self.decoder_embeddings = tf.Variable(
                        tf.constant(
                            0.0,
                            shape=(self.target_vocab_size,
                                   self.embedding_size)
                        ),
                        trainable=True,
                        name='embeddings'
                    )

                    self.decoder_embeddings_placeholder = tf.placeholder(
                        dtype=tf.float32,
                        shape=(self.target_vocab_size, self.embedding_size),
                    )

                    self.decoder_embeddings_init = self.decoder_embeddings.assign(self.decoder_embeddings_placeholder)
                    #运行时通过placeholder传入embedding matrix，通过assign的形式进行赋值。
                ###是否加载预先训练好的embedding matrix###
                else:
                    self.decoder_embeddings = tf.get_variable(
                        name='embeddings',
                        shape=(self.target_vocab_size, self.embedding_size),
                        initializer=self.initializer,
                        dtype=tf.float32
                    )
                    #不加载预先训练好的embedding matrix的情况，声明一个decoder_embeddings matrix,使用-0.5到0.5的均匀分布进行初始化
                ##############################################加载解码器的embedding##################################################

            # 定义输出的projection，实际上就是全连接层
            ##################################解码器的映射############################################
            self.decoder_output_projection = layers.Dense(
                self.target_vocab_size,
                dtype=tf.float32,
                use_bias=False,
                name='decoder_output_projection'
            )
            ##################################解码器的映射############################################
            start_tokens = tf.tile(
                [WordSequence.START],
                [self.batch_size]
            )
            end_token = WordSequence.END

            def embed_and_input_proj(inputs):
                return tf.nn.embedding_lookup(
                    self.decoder_embeddings,
                    inputs
                )
            ################################### 没有使用beamsearch的情况 ################################
            if not self.use_beamsearch_decode:
                decoding_helper = seq2seq.GreedyEmbeddingHelper(
                    start_tokens=start_tokens,
                    end_token=end_token,
                    embedding=embed_and_input_proj #这里embedding参数作用是获得embedding vector的id
                )

            #这个时候使用的decoding_helper 就是贪婪模式下的Helper
                '''
                对output使用argmax(treated as logits)并且送入到embedding matrix中查询embedding vector，
                得到下一个输入值

             '''

                inference_decoder = seq2seq.BasicDecoder(
                    cell=self.decoder_cell,
                    helper=decoding_helper,
                    initial_state=self.decoder_initial_state,
                    output_layer=self.decoder_output_projection
                )
            ################################### 没有使用beamsearch的情况 ################################

            else:

                ##################################beamsearch 的inference_decoder##############################################
                # 这里的BeamSearchDecoder 传入的initial_state是经过变换，成了原来的beam_width 这么多倍。
                inference_decoder = BeamSearchDecoder(
                    cell=self.decoder_cell,
                    embedding=embed_and_input_proj,
                    start_tokens=start_tokens,
                    end_token=end_token,
                    initial_state=self.decoder_initial_state,
                    beam_width=self.beam_width,
                    output_layer=self.decoder_output_projection
                )
                ##################################beamsearch 的inference_decoder##############################################



            if self.max_decode_step is not None:
                max_decoder_step = self.max_decode_step
            else:
                max_decoder_step = tf.round(
                    tf.reduce_max(self.encoder_inputs_length) * 4
                )

            ###############################解码开始####################################

            (
                self.decoder_outputs_decode,
                self.final_state,
                final_sequence_lengths
            ) = (seq2seq.dynamic_decode(
                decoder=inference_decoder,
                output_time_major=self.time_major,
                impute_finished=False,
                maximum_iterations=100,
                swap_memory=True,
                scope=decoder_scope
            ))
            ###############################解码开始####################################

            ##############################没有使用beamsearch 解码的情况############################
            if not self.use_beamsearch_decode:
                dod = self.decoder_outputs_decode
                self.decoder_pred_decode = dod.sample_id
                # self.decoder_pred_decode = tf.transpose(
                #     self.decoder_pred_decode, (1, 0)
                # )
                return self.decoder_pred_decode,final_sequence_lengths
            ##############################没有使用beamsearch 解码的情况############################
            else:

            ##############################使用beamsearch 解码的情况############################
                self.decoder_pred_decode = self.decoder_outputs_decode.predicted_ids
                if self.time_major:
                    self.decoder_pred_decode = tf.transpose(
                        self.decoder_pred_decode, (1, 0, 2)
                    )

                self.decoder_pred_decode = tf.transpose(self.decoder_pred_decode,(0,2,1))

                dod = self.decoder_outputs_decode
                self.beam_prob = dod.beam_search_decoder_output.scores
            ##############################使用beamsearch 解码的情况############################
                return self.decoder_pred_decode,self.beam_prob,final_sequence_lengths






    def build_predict(self,encoder_input, encoder_input_size):
        self.encoder_inputs = encoder_input
        self.encoder_inputs_length = encoder_input_size
        encoder_output,encoder_state = self.build_encoder()
        result = self.predict(encoder_output=encoder_output,encoder_state=encoder_state)
        return result


    def save(self,sess,save_path='model.ckpt',index=None):
        '''
        在TensorFlow里，保存模型的格式有两种：
        ckpt：训练模型后的保存，这里面会保存所有的训练参数，文件相对来讲比较大，可以用来进行模型的恢复和加载
        pb：用于模型的最后上线部署，这里面的线上部署指的是TensorFlow Serving进行模型的发布，一般发布成grpc形式
        的接口
        :param sess:
        :param save_path:
        :return:
        '''

        self.saver.save(sess,save_path=save_path+'-'+str(index))

    def load(self,sess,save_path='model.ckpt'):
        print('try load model from',save_path)
        self.saver.restore(sess,save_path)







if __name__ == '__main__':
    # dataset = make_src_trg_dataset(src_path=en_corpors_path,trg_path=zh_corpors_path,batch_size=100)
    # dataset = dataset.shuffle(10000)
    # iterator = dataset.make_initializable_iterator()
    # with tf.Session() as sess:
    # #     for i in range(1000):
    # #         datas = an_iterator.get_next()
    # #         result = sess.run(datas)
    # #         (encoder_inputs, encoder_inputs_size), (decoder_input, decoder_input_label, decoder_input_size) = result
    # #         print(encoder_inputs_size)
    #     sess.run(iterator.initializer)
    #
    #     while True:
    #         datas = iterator.get_next()
    #         result = sess.run(datas)
    #         (encoder_inputs, encoder_inputs_size), (decoder_input, decoder_input_label, decoder_input_size) = result
    #         print(encoder_inputs_size.shape)
    #         print(batch_size)
    #         assert encoder_inputs_size.shape[0] == batch_size,'sequence_length 的shape应该和batchsize的一致'


    en_ws = pickle.load(open('./datas/corpors_en.pkl','rb'))
    print('en_ws',len(en_ws.dict))
    zh_ws = pickle.load(open('./datas/corpors_zh.pkl','rb'))
    print('zh_ws',len(zh_ws.dict))





