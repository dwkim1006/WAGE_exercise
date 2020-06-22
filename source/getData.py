# coding=utf-8
import tensorflow as tf
import numpy as np
import Option

def preprocess(x, train=False):
  dataSet = Option.dataSet
  if dataSet == 'CIFAR10':
    if train:  #dataset==CIFAR10 & train=True 일때 아래 조건들로 수행능력 향상
      x = tf.image.resize_image_with_crop_or_pad(x, 40, 40)  # 각 면에 4 pixel씩 패딩 추가
      x = tf.random_crop(x, [32, 32, 3]) # x를 [32,32,3]로 다시 축소(Crop)
      x = tf.image.random_flip_left_right(x) #좌우반전 시키기(flip image horizontally)
  else:
    print 'Unkown dataset',dataSet,'no preprocess'
  x = tf.transpose(x, [2, 0, 1]) # from HWC[32,32,3] to CHW[3,32,32]
  return x


def loadData(dataSet,batchSize,numThread):

  pathNPZ = '../dataSet/' + dataSet + '.npz'    # ../dataSet/CIFAR10.npz 을 경로
  numpyTrainX, numpyTrainY, numpyTestX, numpyTestY, label = loadNPZ(pathNPZ) #numpyTrainX를 알기 위해 아래의 loadNPZ를 알자
  numTrain = numpyTrainX.shape[0] # numpyTrainX 튜플의 첫번째 요소 : 50000
  numTest = numpyTestX.shape[0]  # numpyTestX 튜플의 첫번째 요소 : 10000

  trainX, trainY = data2Queue(numpyTrainX, numpyTrainY, batchSize,numThread, True,True)  # data2Queue를 알기 위해 아래의 인스턴스를 보자
  testX, testY = data2Queue(numpyTestX, numpyTestY, 100, 1, False,False) #dataX = numpyTrainX, dataY = numpyTrainY

  return trainX,trainY,testX,testY,numTrain,numTest,label


# get dataset from NPZ files
def loadNPZ(pathNPZ):
  data = np.load(pathNPZ)

  trainX = data['trainX']  # CIFAR10.npz 내의 data['trainX']를 np.zeros[50000,(32,32,3)]에 대입
  trainY = data['trainY']

  testX = data['testX']
  testY = data['testY']

  label = data['label']
  return trainX, trainY, testX, testY, label

  # 데이터를 모델에 Feed할때 데이터의 양이 많아서 메모리에 모두 적재할 수 없고 큐를 통해 읽어드리면서 학습(FIFO)
def data2Queue(dataX, dataY, batchSize, numThreads, shuffle=False, isTraining=True, seed=None):
  q = tf.FIFOQueue(capacity=dataX.shape[0], dtypes=[dataX.dtype, dataY.dtype],shapes=[dataX.shape[1:],dataY.shape[1:]])
  # 큐 길이는 dataX.shape[0]인 50000, dataX = (50000,32,32,3)? -> 클수록 메모리사용량은 많아지지만 스레드가 채우길 기다리는 시간 감
  enqueue_op = q.enqueue_many([dataX, dataY])  # 한번에 dataX와 dataY를 qRunner를 통해 queue에 enqueue(큐 뒤쪽에 항목을 삽입)한다.
  sampleX, sampleY = q.dequeue()  # dequeue(큐 앞쪽의 항목을 반환하고 제거)
  qRunner = tf.train.QueueRunner(q, [enqueue_op])  # [enqueue_op]를 하나만 적었기에 num_thread = 1 ?
  tf.train.add_queue_runner(qRunner)

  sampleX_ = preprocess(sampleX, isTraining)  # sampleX를 preprocess하는 isTraining = True이기 때문에 위쪽의 세가지 과정을 수행.
  #sampleX_를 매개변수로 주면서 혹시 있을 명령어 sampleX와의 충돌을 피하기 위해 underscore 사용.
  if shuffle:  #Shuffle=False
    batchX, batchY = tf.train.shuffle_batch([sampleX_, sampleY],
                                            batch_size=batchSize,
                                            num_threads=numThreads, capacity=dataX.shape[0],
                                            min_after_dequeue=dataX.shape[0] / 2,
                                            #dequeue하고 남는 queue의 최소 element number이며 element를 섞는 정도를 위해 존
                                            #min_after_dequeue와 capacity 사이의 요소를 읽어 메모리에 저장하고 무작위로 배치
                                            #커질수록 더 많은 요소가 무작위적으로 batch됨
                                            seed=seed)
  else:
    batchX, batchY = tf.train.batch([sampleX_, sampleY],  # 입력tensor가 (*, x,y,z)이면 output은 (batchsize,x,y,z)
                                    batch_size=batchSize,
                                    num_threads=numThreads,
                                    capacity=dataX.shape[0])

  return batchX, batchY