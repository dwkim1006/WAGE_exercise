# coding=utf-8
import numpy as np
import time
import tensorflow as tf
import NN
import Option
import Log
import getData
import Quantize
from tqdm import tqdm

# for single GPU quanzation
def quantizeGrads(Grad_and_vars):
  if Quantize.bitsG <= 16:  #bitsG = 8
    grads = []
    for grad_and_vars in Grad_and_vars:
      grads.append([Quantize.G(grad_and_vars[0]), grad_and_vars[1]])  # grad_and_vars[0],grad_and_vars[1]]?
    print(Grad_and_vars)
    print("이 부분")
    return grads
  return Grad_and_vars

def showVariable(keywords=None):
  Vars = tf.global_variables()  # 지금까지 정의된 변수 확인.
  Vars_key = []
  for var in Vars:
    print var.device,var.name,var.shape,var.dtype
    if keywords is not None:  # if x = if x is not none
      if var.name.lower().find(keywords) > -1: # var에 keyword란 이름이 들어있으면 Vars_key에 var 추
        Vars_key.append(var)
    else:
      Vars_key.append(var)   # keywords = None
  print(Vars_key)
  print("여기야")
  return Vars_key

def main():
  # get Option
  GPU = Option.GPU
  batchSize = Option.batchSize  #  batchSize = 128
  pathLog = '../log/' + Option.Time + '(' + Option.Notes + ')' + '.txt'
  Log.Log(pathLog, 'w+', 1) # set log file
  print time.strftime('%Y-%m-%d %X', time.localtime()), '\n'
  print open('Option.py').read()

  # get data
  numThread = 4*len(GPU)         # len(GPU) = 1
  assert batchSize % len(GPU) == 0, ('batchSize must be divisible by number of GPUs')  # 128 = 32 * 4
  with tf.device('/cpu:0'):  # 첫번째 CPU사용.
    batchTrainX,batchTrainY,batchTestX,batchTestY,numTrain,numTest,label =\
        getData.loadData(Option.dataSet,batchSize,numThread)

  batchNumTrain = numTrain / batchSize   # batchSize = 128, numTrain = 
  batchNumTest = numTest / 100

  optimizer = Option.optimizer
  # global_step이란 변수를 0으로 초기화하여 생성.
  global_step = tf.get_variable('global_step', [], dtype=tf.int32, initializer=tf.constant_initializer(0), trainable=False)
  Net = []

  # CPU와 GPU를 동시에 지원하게하면 GPU에 먼저 배치.
  # on my machine, alexnet does not fit multi-GPU training
  # for single GPU
  with tf.device('/gpu:%d' % GPU[0]):  # GPU[0] = 0
    Net.append(NN.NN(batchTrainX, batchTrainY, training=True, global_step=global_step))
    lossTrainBatch, errorTrainBatch = Net[-1].build_graph()
    update_op = tf.get_collection(tf.GraphKeys.UPDATE_OPS)  # batchnorm moving average update ops (not used now)
    # batchnorm에서 test할때 사용할 moving_average와 moving_var이 train할때 계산을 해줘야하는데,
    # train과정에서 moving_average와 moving_var이 직접 호출되지 않기에 train 하는것과 별도로
    # 이것들에 대한 op를 실행해줘서 업데이트를 해줘야한다(갱신연산)

    # since we quantize W at the beginning and the update delta_W is quantized,
    # there is no need to quantize W every iteration
    # we just clip W after each iteration for speed
    update_op += Net[0].W_clip_op

    gradTrainBatch = optimizer.compute_gradients(lossTrainBatch)

    gradTrainBatch_quantize = quantizeGrads(gradTrainBatch)
    #train_op update_ops 연산이 끝난 후에 연산되게 함.
    with tf.control_dependencies(update_op):
      train_op = optimizer.apply_gradients(gradTrainBatch_quantize, global_step=global_step)

    tf.get_variable_scope().reuse_variables()  # 변수를 재사용하기 위한 플래그
    Net.append(NN.NN(batchTestX, batchTestY, training=False))
    _, errorTestBatch = Net[-1].build_graph()



  showVariable()
  # 변수 생성,초기화,저장.
  # Build an initialization operation to run below.
  config = tf.ConfigProto()
  config.gpu_options.allow_growth = True  # GPU 메모리 증가 허용(요구되는 만큼의 GPU만 할당)
  config.allow_soft_placement = True  # 명시된 디바이스가 없을경우 TF가 자동으로 디바이스 선택.
  config.log_device_placement = False  # 하지만 디바이스 명시하지 않음
  sess = Option.sess = tf.InteractiveSession(config=config)
  sess.run(tf.global_variables_initializer())
  saver = tf.train.Saver(max_to_keep=None)  # 최대 모델 수 정하지 않고 저장.
  # Start the queue runners.
  tf.train.start_queue_runners(sess=sess)



  def getErrorTest():
    errorTest = 0.
    for i in tqdm(xrange(batchNumTest),desc = 'Test', leave=False):
      errorTest += sess.run([errorTestBatch])[0]
    errorTest /= batchNumTest
    return errorTest

  if Option.loadModel is not None:
    print 'Loading model from %s ...' % Option.loadModel,
    saver.restore(sess, Option.loadModel)
    print 'Finished',
    errorTestBest = getErrorTest()
    print 'Test:', errorTestBest

  else:  # True이므로
   # at the beginning, we discrete W
    sess.run([Net[0].W_q_op])

  print "\nOptimization Start!\n"
  for epoch in xrange(1000):
    # check lr_schedule
    if len(Option.lr_schedule) / 2:
      if epoch == Option.lr_schedule[0]: #epoch이 0이면 [0]을 뽑아내고
        Option.lr_schedule.pop(0)
        lr_new = Option.lr_schedule.pop(0)
        if lr_new == 0:
          print 'Optimization Ended!'  #
          exit(0)
        lr_old = sess.run(Option.lr)
        sess.run(Option.lr.assign(lr_new))
        print 'lr: %f -> %f' % (lr_old, lr_new)

    print 'Epoch: %03d ' % (epoch),


    lossTotal = 0.
    errorTotal = 0
    t0 = time.time()
    for batchNum in tqdm(xrange(batchNumTrain), desc='Epoch: %03d' % epoch, leave=False, smoothing=0.1):
      if Option.debug is False:
        _, loss_delta, error_delta = sess.run([train_op, lossTrainBatch, errorTrainBatch])
      else:
        _, loss_delta, error_delta, H, W, W_q, gradH, gradW, gradW_q=\
        sess.run([train_op, lossTrainBatch, errorTrainBatch, Net[0].H, Net[0].W, Net[0].W_q, Net[0].gradsH, Net[0].gradsW, gradTrainBatch_quantize])

      lossTotal += loss_delta
      errorTotal += error_delta

    lossTotal /= batchNumTrain
    errorTotal /= batchNumTrain

    print 'Loss: %.4f Train: %.4f' % (lossTotal, errorTotal),

    # get test error
    errorTest = getErrorTest()
    print 'Test: %.4f FPS: %d' % (errorTest, numTrain / (time.time() - t0)),

    if epoch == 0:
      errorTestBest = errorTest
    if errorTest < errorTestBest:
      if Option.saveModel is not None:
        saver.save(sess, Option.saveModel)
        print 'S',
    if errorTest < errorTestBest:
      errorTestBest = errorTest
      print 'BEST',

    print ''


if __name__ == '__main__':
  main()

