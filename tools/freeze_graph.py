from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import _init_paths
from model.config import cfg


import _init_paths
import tensorflow as tf
from nets.inception_v1 import inception_v1
from nets.vgg16 import vgg16
from nets.squeeze import squeeze


cfg.POOLING_SIZE=3
cfg.ANCHOR_SCALES=[8,16,32]

#output_dir = 'output/vgg16/lisa_tl_train/default/'
#output_dir = 'output/inception_v1/lisa_tl_train/default/'
output_dir = 'output/squeeze/carback/default/'

sess = tf.Session()

with sess.graph.as_default():
    #net = vgg16(batch_size=1)
    #net = inception_v1(batch_size=1)
    net = squeeze(batch_size=1)
    net.create_architecture(sess, 'TEST', 2, tag='default', anchor_scales=cfg.ANCHOR_SCALES)
    saver = tf.train.Saver()
    #saver.restore(sess, output_dir + 'vgg16_faster_rcnn_iter_70000.ckpt')
    #saver.restore(sess, output_dir + 'inception_v1_faster_rcnn_iter_20000.ckpt')
    saver.restore(sess, output_dir + 'squeeze_faster_rcnn_iter_10000.ckpt')
    output_graph_def = tf.graph_util.convert_variables_to_constants(sess, sess.graph.as_graph_def(), ['squeeze/rpn_cls_prob/transpose_1',
                                                                                                          'squeeze/rpn_bbox_pred/BiasAdd',
                                                                                                          'squeeze/cls_prob',
                                                                                                          'squeeze/bbox_pred'])
    #output_graph_def = tf.graph_util.convert_variables_to_constants(sess, sess.graph.as_graph_def(), ['InceptionV1/rpn_cls_prob/transpose_1',
    #                                                                                                      'InceptionV1/rpn_bbox_pred/BiasAdd',
    #                                                                                                      'InceptionV1/cls_prob',
    #                                                                                                      'InceptionV1/bbox_pred/BiasAdd'])

    with tf.gfile.FastGFile(output_dir + 'carback_output_graph.pb', 'wb') as f:
      f.write(output_graph_def.SerializeToString())

