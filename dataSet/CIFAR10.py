# coding=utf-8
import cPickle
import numpy as np
import os
import tarfile
from six.moves import urllib

URL = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
"""Download the cifar 100 dataset."""
if not os.path.exists('cifar-10-python.tar.gz'):  # os.path.exists(path) path가 존재하면 True를 리턴
    print("Downloading...")  # 존재하지 않아서 False일때 실행 : URL에서 파일 Download
    urllib.request.urlretrieve(URL, 'cifar-10-python.tar.gz')
    # urllib.request : url주소의 문서를 열고 읽기 위해 사용
    # urlretrieve : URL로 표시된 네트워크객체(URL주소의 문서)를 파일로 저장해 직접 다운로드, cifar-10-python.tar.gz의 이름으로 파일 저장.

if not os.path.exists('cifar-10-batches-py/test_batch'):  # 압축이 안풀려 cifar-10-batches-py/test_batch 파일이 없다고 판단되면
    print("Extracting files...")  # 파일 압축해제
    tar = tarfile.open('cifar-10-python.tar.gz')  # 압축을 풀때는 tarfile, 압축 풀 파일 지정(cifar-10-python.tar.gz)후 열기.
    tar.extractall()  # 압축 풀고 싶은 경로, 빈칸일 경우 현재 경로에 압축 풀기
    tar.close()  # 압축파일 닫기

# 0으로 된 빈 배열을 만드는데 왜만드냐면 reshape 후의 training data를 담기위해(hold)
trainX = np.zeros([50000, 32, 32, 3], dtype=np.uint8)  # u(0을 포함한 양수) + int8(2^8) : 0~255
trainY = np.zeros([50000, 10], dtype=np.uint8)  # 50000개의 input image array(32,32,3)가 10개의 label을 가질 수 있음
testX = np.zeros([10000, 32, 32, 3], dtype=np.uint8)
testY = np.zeros([10000, 10], dtype=np.uint8)
label = ['airplane', 'automoblie', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']  # 10개의 label(1,10)

trainFileName = ['cifar-10-batches-py/data_batch_1',
                 'cifar-10-batches-py/data_batch_2',
                 'cifar-10-batches-py/data_batch_3',
                 'cifar-10-batches-py/data_batch_4',
                 'cifar-10-batches-py/data_batch_5']

testFileName = ['cifar-10-batches-py/test_batch']
# 파일이 어떤 모양으로 저장되어있는지 visualization도 같이해보기.

# batch_data 하나 당 (10000개의 샘플데이터,32,32,3)의 배열을 가짐
index = 0
for name in trainFileName:
    f = open(name, 'rb')
    dict = cPickle.load(f)  # 위에서 CIFAR10을 다운로드 받으면 사실 사진이 아니라 binary파일로 되어있다.
    # pickle.dump는 객체의 데이터를 쭉 bin으로  직렬화, .load는 직렬화된 바이트나 파일을 원래의 객체로 복원(역직렬화)
    # 저 bin파일의 구조는 Label + 3072픽셀(R1024 + G1024 + B1024)이 batch파일 하나당 10000개씩
    f.close()
    trainX[index:index + 10000, ...] = dict['data'].reshape([10000, 3, 32, 32]).transpose([0, 2, 3, 1])
    # dict의 'data'는 (10000, 3072pixel)의 모양이고 이걸 reshape를 통해 (10000,3,32,32), transpose로 (10000,32,32,3)
    # 위 과정을 통해 RGB의 form을 가짐.
    # Transpose : [R1 ...R1024],[G1...G1024],[B1...B1024] -> [R1,G1,B1],[R2,G2,B2]..[R1024,G1024,B1024]
    # CNN의 conv2d는 NHWC를 input tensor로 받는데(by tensorflow.org) 여기선 NCHW로 받음(getData.py, NN.py)
    # input tensor가 WHC or CWH 여야 한다?
    trainY[np.arange(index, index + 10000), dict['labels']] = 1
    index += 10000

index = 0
for name in testFileName:
    f = open(name, 'rb')
    dict = cPickle.load(f)
    f.close()
    testX[index:index + 10000, ...] = dict['data'].reshape([10000, 3, 32, 32]).transpose([0, 2, 3, 1])
    testY[np.arange(index, index + 10000), dict['labels']] = 1
    index += 10000

np.savez('CIFAR10.npz', trainX=trainX, trainY=trainY, testX=testX, testY=testY, label=label)
# 여러개의 배열을 1개의 압축되지 않은 .npz포맷 파일로 저장하기
print 'dataset saved'
