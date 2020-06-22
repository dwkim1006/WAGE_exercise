# coding=utf-8
import tensorflow as tf
import Option

LR    = Option.lr
bitsW = Option.bitsW
bitsA = Option.bitsA
bitsG = Option.bitsG  #bitsG = 8
bitsE = Option.bitsE
bitsR = Option.bitsR
L2 = Option.L2

Graph = tf.get_default_graph()

def S(bits):
  return 2.0 ** (bits - 1)

def Shift(x):    #  2 ^ ((log2의 x)의 반올림값)
  return 2 ** tf.round(tf.log(x) / tf.log(2.0))

def C(x, bits=32):
  if bits > 15 or bits == 1:
    delta = 0.
  else:
    delta = 1. / S(bits)
  MAX = +1 - delta
  MIN = -1 + delta
  x = tf.clip_by_value(x, MIN, MAX, name='saturate')
  return x

def Q(x, bits):
  if bits > 15:
    return x
  elif bits == 1:  # BNN(banarized)
    return tf.sign(x)
  else:
    SCALE = S(bits)
    return tf.round(x * SCALE) / SCALE

def W(x,scale = 1.0):
  with tf.name_scope('QW'):
    y = Q(C(x, bitsW), bitsW)
    # we scale W in QW rather than QA for simplicity
    if scale > 1.8:
      y = y/scale
    # if bitsG > 15:
      # when not quantize gradient, we should rescale the scale factor in backprop
      # otherwise the learning rate will have decay factor scale
      # x = x * scale
    return x + tf.stop_gradient(y - x)  # skip derivation of Quantize and Clip

def A(x):
  with tf.name_scope('QA'):
    x = C(x, bitsA)
    y = Q(x, bitsA)
    return x + tf.stop_gradient(y - x)  # skip derivation of Quantize, but keep Clip

def G(x):
  with tf.name_scope('QG'):  # 변수의 이름을 QG라는 그룹으로 묶어서 계층화.
    if bitsG > 15:  # bitsG = 8
      return x
    else:
      if x.name.lower().find('batchnorm') > -1:   # x의 이름 소문자로 만들고 x에서 batchnorm이라는 문자를 찾게된다면(못찾으면 -1)
        return x  # batch norm parameters, not quantize now

      xmax = tf.reduce_max(tf.abs(x))  # x에 절대값 씌우고 지정된 차원을 따라 최댓값을 선택.
      x = x / Shift(xmax)  #  x / 2^ (log(xmax-2.0)의 반올림값)

      norm = Q(LR * x, bitsR)  #

      norm_sign = tf.sign(norm)
      norm_abs = tf.abs(norm)
      norm_int = tf.floor(norm_abs)
      norm_float = norm_abs - norm_int
      rand_float = tf.random_uniform(x.get_shape(), 0, 1)
      norm = norm_sign * ( norm_int + 0.5 * (tf.sign(norm_float - rand_float) + 1) )

      return norm / S(bitsG)

@tf.RegisterGradient('Error')
def error(op, x):
  if bitsE > 15:
    return x
  else:
    xmax = tf.reduce_max(tf.abs(x))
    xmax_shift = Shift(xmax)
    return Q(C( x /xmax_shift, bitsE), bitsE)

def E(x):
  with tf.name_scope('QE'):
    with Graph.gradient_override_map({'Identity': 'Error'}):
      return tf.identity(x)

