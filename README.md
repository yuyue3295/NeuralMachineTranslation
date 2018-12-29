### NeuralMachineTranslation
#基于seq2seq的神经网络翻译
  本项目的任务是训练英文翻译成中文的神经网络模型，改自慕课网和 《TensorFlow 实战Google深度学习框架》,模型文件的连接是https://pan.baidu.com/s/1ftRHrLzZx008WaacpV0JfQ
，参考train_test.py文件中定义的目录saver.restore(sess=sess,save_path='./model/nmt.ckpt-51')，将nmt.ckpt.meta,nmt.ckpt.index,nmt.ckpt.data三个文件放入到项目文件目录
的model文件夹下。
  运行train_test.py 启动神经网络机器翻译程序。
  
下面是部分翻译的结果：<br>
on the united nations ' role , wang guangya emphatically expounded the chinese government 's stand that it hopes the united nations will play a more active role in peacefully settling international disputes .<br>
在联合国的作用中,王光亚强调了中国政府希望联合国在和平解决国际争端中发挥更加积极的作用.</S><br>
在联合国的作用时,王光亚强调了中国政府希望联合国在和平解决国际争端中发挥更加积极的作用.</S><br>
在联合国的作用中,王光亚强调了中国政府希望联合国在和平解决国际争端中发挥更加积极作用.</S><br>
在联合国的作用中,王光亚强调了中国政府希望联合国在和平解决国际争端中发挥更加积极作用的立场.<br>
he pointed out : the various norms governing international relations , including the five principles of peaceful coexistence , established after wwii meet the requirements of historical development and progress of times .<br>
他指出,战后包括和平共处五项原则在内的国际关系的国际关系符合时代发展与进步的要求.</S><br>
他指出,战后包括和平共处五项原则在内的国际关系的国际关系符合时代的发展与进步的要求.</S><br>
他指出,战后包括和平共处五项原则在内的国际关系准则,创立了时代的历史发展与进步的要求.</S><br>
他指出,战后包括和平共处五项原则在内的国际关系的不同准则,初步确立了时代发展与进步的要求.</S><br>

.....
  
