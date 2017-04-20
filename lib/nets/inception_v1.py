#
# by Zhe Zhang
#

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import tensorflow.contrib.slim as slim
from tensorflow.contrib.slim import losses
from tensorflow.contrib.slim import arg_scope


import numpy as np

try:
    import cPickle as pickle
except ImportError:
    import pickle

from layer_utils.snippets import generate_anchors_pre
from layer_utils.proposal_layer import proposal_layer
from layer_utils.proposal_top_layer import proposal_top_layer
from layer_utils.anchor_target_layer import anchor_target_layer
from layer_utils.proposal_target_layer import proposal_target_layer
from nets.network import Network
from model.config import cfg


class inception_v1(Network):
    def __init__(self, batch_size=1):
        Network.__init__(self, batch_size=batch_size, feat_stride=4.0)
        self._arch = 'inception_v1'

    def build_network(self, sess, is_training=True):
        with tf.variable_scope('InceptionV1', 'InceptionV1',
                                regularizer=tf.contrib.layers.l2_regularizer(cfg.TRAIN.WEIGHT_DECAY)):
            if cfg.TRAIN.TRUNCATED:
                initializer = tf.truncated_normal_initializer(mean=0.0, stddev=0.01)
                initializer_bbox = tf.truncated_normal_initializer(mean=0.0, stddev=0.001)
            else:
                initializer = tf.random_normal_initializer(mean=0.0, stddev=0.01)
                initializer_bbox = tf.random_normal_initializer(mean=0.0, stddev=0.001)
            with arg_scope([slim.conv2d, slim.max_pool2d], stride=1, padding='SAME'):
                net = slim.conv2d(self._image, 64, [7, 7], stride=2, scope='Conv2d_1a_7x7')
                net = slim.max_pool2d(net, [3, 3], stride=2, scope='MaxPool_2a_3x3')
                net = slim.conv2d(net, 64, [1, 1], scope='Conv2d_2b_1x1')
                net = slim.conv2d(net, 192, [3, 3], scope='Conv2d_2c_3x3')
                #net = slim.max_pool2d(net, [3, 3], stride=2, padding='SAME', scope='MaxPool_3a_3x3')

                #with tf.variable_scope('Mixed_3b'):
                #  with tf.variable_scope('Branch_0'):
                #    branch_0 = slim.conv2d(net, 64, [1, 1], scope='Conv2d_0a_1x1')
                #  with tf.variable_scope('Branch_1'):
                #    branch_1 = slim.conv2d(net, 96, [1, 1], scope='Conv2d_0a_1x1')
                #    branch_1 = slim.conv2d(branch_1, 128, [3, 3], scope='Conv2d_0b_3x3')
                #  with tf.variable_scope('Branch_2'):
                #    branch_2 = slim.conv2d(net, 16, [1, 1], scope='Conv2d_0a_1x1')
                #    branch_2 = slim.conv2d(branch_2, 32, [3, 3], scope='Conv2d_0b_3x3')
                #  with tf.variable_scope('Branch_3'):
                #    branch_3 = slim.max_pool2d(net, [3, 3], stride=1, padding='SAME', scope='MaxPool_0a_3x3')
                #    branch_3 = slim.conv2d(branch_3, 32, [1, 1], scope='Conv2d_0b_1x1')
                #  net = tf.concat([branch_0, branch_1, branch_2, branch_3], 3)

                #with tf.variable_scope('Mixed_3c'):
                #  with tf.variable_scope('Branch_0'):
                #    branch_0 = slim.conv2d(net, 128, [1, 1], scope='Conv2d_0a_1x1')
                #  with tf.variable_scope('Branch_1'):
                #    branch_1 = slim.conv2d(net, 128, [1, 1], scope='Conv2d_0a_1x1')
                #    branch_1 = slim.conv2d(branch_1, 192, [3, 3], scope='Conv2d_0b_3x3')
                #  with tf.variable_scope('Branch_2'):
                #    branch_2 = slim.conv2d(net, 32, [1, 1], scope='Conv2d_0a_1x1')
                #    branch_2 = slim.conv2d(branch_2, 96, [3, 3], scope='Conv2d_0b_3x3')
                #  with tf.variable_scope('Branch_3'):
                #    branch_3 = slim.max_pool2d(net, [3, 3], stride=1, padding='SAME', scope='MaxPool_0a_3x3')
                #    branch_3 = slim.conv2d(branch_3, 64, [1, 1], scope='Conv2d_0b_1x1')
                #  net = tf.concat([branch_0, branch_1, branch_2, branch_3], 3)

                #net = slim.max_pool2d(net, [3, 3], stride=2, padding='SAME', scope='MaxPool_4a_3x3')

                #with tf.variable_scope('Mixed_4b'):
                #  with tf.variable_scope('Branch_0'):
                #    branch_0 = slim.conv2d(net, 192, [1, 1], scope='Conv2d_0a_1x1')
                #  with tf.variable_scope('Branch_1'):
                #    branch_1 = slim.conv2d(net, 96, [1, 1], scope='Conv2d_0a_1x1')
                #    branch_1 = slim.conv2d(branch_1, 208, [3, 3], scope='Conv2d_0b_3x3')
                #  with tf.variable_scope('Branch_2'):
                #    branch_2 = slim.conv2d(net, 16, [1, 1], scope='Conv2d_0a_1x1')
                #    branch_2 = slim.conv2d(branch_2, 48, [3, 3], scope='Conv2d_0b_3x3')
                #  with tf.variable_scope('Branch_3'):
                #    branch_3 = slim.max_pool2d(net, [3, 3], stride=1, padding='SAME', scope='MaxPool_0a_3x3')
                #    branch_3 = slim.conv2d(branch_3, 64, [1, 1], scope='Conv2d_0b_1x1')
                #  net = tf.concat([branch_0, branch_1, branch_2, branch_3], 3)

                #with tf.variable_scope('Mixed_4c'):
                #  with tf.variable_scope('Branch_0'):
                #    branch_0 = slim.conv2d(net, 160, [1, 1], scope='Conv2d_0a_1x1')
                #  with tf.variable_scope('Branch_1'):
                #    branch_1 = slim.conv2d(net, 112, [1, 1], scope='Conv2d_0a_1x1')
                #    branch_1 = slim.conv2d(branch_1, 224, [3, 3], scope='Conv2d_0b_3x3')
                #  with tf.variable_scope('Branch_2'):
                #    branch_2 = slim.conv2d(net, 24, [1, 1], scope='Conv2d_0a_1x1')
                #    branch_2 = slim.conv2d(branch_2, 64, [3, 3], scope='Conv2d_0b_3x3')
                #  with tf.variable_scope('Branch_3'):
                #    branch_3 = slim.max_pool2d(net, [3, 3], stride=1, padding='SAME', scope='MaxPool_0a_3x3')
                #    branch_3 = slim.conv2d(branch_3, 64, [1, 1], scope='Conv2d_0b_1x1')
                #  net = tf.concat([branch_0, branch_1, branch_2, branch_3], 3)

                #with tf.variable_scope('Mixed_4d'):
                #  with tf.variable_scope('Branch_0'):
                #    branch_0 = slim.conv2d(net, 128, [1, 1], scope='Conv2d_0a_1x1')
                #  with tf.variable_scope('Branch_1'):
                #    branch_1 = slim.conv2d(net, 128, [1, 1], scope='Conv2d_0a_1x1')
                #    branch_1 = slim.conv2d(branch_1, 256, [3, 3], scope='Conv2d_0b_3x3')
                #  with tf.variable_scope('Branch_2'):
                #    branch_2 = slim.conv2d(net, 24, [1, 1], scope='Conv2d_0a_1x1')
                #    branch_2 = slim.conv2d(branch_2, 64, [3, 3], scope='Conv2d_0b_3x3')
                #  with tf.variable_scope('Branch_3'):
                #    branch_3 = slim.max_pool2d(net, [3, 3], stride=1, padding='SAME', scope='MaxPool_0a_3x3')
                #    branch_3 = slim.conv2d(branch_3, 64, [1, 1], scope='Conv2d_0b_1x1')
                #  net = tf.concat([branch_0, branch_1, branch_2, branch_3], 3)

                #with tf.variable_scope('Mixed_4e'):
                #  with tf.variable_scope('Branch_0'):
                #    branch_0 = slim.conv2d(net, 112, [1, 1], scope='Conv2d_0a_1x1')
                #  with tf.variable_scope('Branch_1'):
                #    branch_1 = slim.conv2d(net, 144, [1, 1], scope='Conv2d_0a_1x1')
                #    branch_1 = slim.conv2d(branch_1, 288, [3, 3], scope='Conv2d_0b_3x3')
                #  with tf.variable_scope('Branch_2'):
                #    branch_2 = slim.conv2d(net, 32, [1, 1], scope='Conv2d_0a_1x1')
                #    branch_2 = slim.conv2d(branch_2, 64, [3, 3], scope='Conv2d_0b_3x3')
                #  with tf.variable_scope('Branch_3'):
                #    branch_3 = slim.max_pool2d(net, [3, 3], stride=1, padding='SAME', scope='MaxPool_0a_3x3')
                #    branch_3 = slim.conv2d(branch_3, 64, [1, 1], scope='Conv2d_0b_1x1')
                #  net = tf.concat([branch_0, branch_1, branch_2, branch_3], 3)

                #with tf.variable_scope('Mixed_4f'):
                #  with tf.variable_scope('Branch_0'):
                #    branch_0 = slim.conv2d(net, 256, [1, 1], scope='Conv2d_0a_1x1')
                #  with tf.variable_scope('Branch_1'):
                #    branch_1 = slim.conv2d(net, 160, [1, 1], scope='Conv2d_0a_1x1')
                #    branch_1 = slim.conv2d(branch_1, 320, [3, 3], scope='Conv2d_0b_3x3')
                #  with tf.variable_scope('Branch_2'):
                #    branch_2 = slim.conv2d(net, 32, [1, 1], scope='Conv2d_0a_1x1')
                #    branch_2 = slim.conv2d(branch_2, 128, [3, 3], scope='Conv2d_0b_3x3')
                #  with tf.variable_scope('Branch_3'):
                #    branch_3 = slim.max_pool2d(net, [3, 3], stride=1, padding='SAME', scope='MaxPool_0a_3x3')
                #    branch_3 = slim.conv2d(branch_3, 128, [1, 1], scope='Conv2d_0b_1x1')
                #  net = tf.concat([branch_0, branch_1, branch_2, branch_3], 3)

                #net = slim.max_pool2d(net, [2, 2], stride=2, padding='SAME', scope='MaxPool_5a_2x2')

                #with tf.variable_scope('Mixed_5b'):
                #  with tf.variable_scope('Branch_0'):
                #    branch_0 = slim.conv2d(net, 256, [1, 1], scope='Conv2d_0a_1x1')
                #  with tf.variable_scope('Branch_1'):
                #    branch_1 = slim.conv2d(net, 160, [1, 1], scope='Conv2d_0a_1x1')
                #    branch_1 = slim.conv2d(branch_1, 320, [3, 3], scope='Conv2d_0b_3x3')
                #  with tf.variable_scope('Branch_2'):
                #    branch_2 = slim.conv2d(net, 32, [1, 1], scope='Conv2d_0a_1x1')
                #    branch_2 = slim.conv2d(branch_2, 128, [3, 3], scope='Conv2d_0a_3x3')
                #  with tf.variable_scope('Branch_3'):
                #    branch_3 = slim.max_pool2d(net, [3, 3], stride=1, padding='SAME', scope='MaxPool_0a_3x3')
                #    branch_3 = slim.conv2d(branch_3, 128, [1, 1], scope='Conv2d_0b_1x1')
                #  net = tf.concat([branch_0, branch_1, branch_2, branch_3], 3)

                #with tf.variable_scope('Mixed_5c'):
                #  with tf.variable_scope('Branch_0'):
                #    branch_0 = slim.conv2d(net, 384, [1, 1], scope='Conv2d_0a_1x1')
                #  with tf.variable_scope('Branch_1'):
                #    branch_1 = slim.conv2d(net, 192, [1, 1], scope='Conv2d_0a_1x1')
                #    branch_1 = slim.conv2d(branch_1, 384, [3, 3], scope='Conv2d_0b_3x3')
                #  with tf.variable_scope('Branch_2'):
                #    branch_2 = slim.conv2d(net, 48, [1, 1], scope='Conv2d_0a_1x1')
                #    branch_2 = slim.conv2d(branch_2, 128, [3, 3], scope='Conv2d_0b_3x3')
                #  with tf.variable_scope('Branch_3'):
                #    branch_3 = slim.max_pool2d(net, [3, 3], stride=1, padding='SAME', scope='MaxPool_0a_3x3')
                #    branch_3 = slim.conv2d(branch_3, 128, [1, 1], scope='Conv2d_0b_1x1')
                #  net = tf.concat([branch_0, branch_1, branch_2, branch_3], 3)
                #with tf.gfile.FastGFile(self._pretrained_model, 'rb') as f:
                #    graph_def = tf.GraphDef()
                #    graph_def.ParseFromString(f.read())
                #    self._image, mixed5b_tensor  = tf.import_graph_def(graph_def, name='', return_elements=['input:0', 'mixed4e:0'])

                self._act_summaries.append(net)
                self._anchor_component()

                rpn = slim.conv2d(net, 512, [3, 3], trainable=is_training, weights_initializer=initializer, scope="rpn_conv/3x3")
                self._act_summaries.append(rpn)
                rpn_cls_score = slim.conv2d(rpn, self._num_anchors*2, [1,1], trainable=is_training, weights_initializer=initializer, padding='VALID', activation_fn=None, scope='rpn_cls_score')
                rpn_cls_score_reshape = self._reshape_layer(rpn_cls_score, 2, 'rpn_cls_score_reshape')
                rpn_cls_prob_reshape = self._softmax_layer(rpn_cls_score_reshape, 'rpn_cls_prob_reshape')
                rpn_cls_prob = self._reshape_layer(rpn_cls_prob_reshape, self._num_anchors*2, 'rpn_cls_prob')
                rpn_bbox_pred = slim.conv2d(rpn, self._num_anchors*4, [1,1], trainable=is_training, weights_initializer=initializer, padding='VALID', activation_fn=None, scope='rpn_bbox_pred')
                if is_training:
                    rois, rois_scores = self._proposal_layer(rpn_cls_prob, rpn_bbox_pred, "rois")
                    rpn_labels = self._anchor_target_layer(rpn_cls_prob, "anchor")

                    with tf.control_dependencies([rpn_labels]):
                        rois, _ = self._proposal_target_layer(rois, rois_scores, 'rpn_rois')
                else:
                    if cfg.TEST.MODE == 'nms':
                        rois, _ = self._proposal_layer(rpn_cls_prob, rpn_bbox_pred, "rois")
                    elif cfg.TEST.MODE == 'top':
                        rois, _ = self._proposal_top_layer(rpn_cls_prob, rpn_bbox_pred, "rois")
                    else:
                        raise NotImplementedError
                if cfg.POOLING_MODE == 'crop':
                    pool5 = self._crop_pool_layer(net, 'pool5')
                else:
                    raise NotImplementedError

                pool5_flat = slim.flatten(pool5, scope='flatten')
                fc6 = slim.fully_connected(pool5_flat, 1024, scope='fc6')
                if is_training:
                    fc6 = slim.dropout(fc6, scope='dropout6')
                fc7 = slim.fully_connected(fc6, 512, scope='fc7')
                if is_training:
                    fc7 = slim.dropout(fc7, scope='dropout7')
                cls_score = slim.fully_connected(fc7, self._num_classes, weights_initializer=initializer, trainable=is_training, activation_fn=None, scope='cls_score')
                cls_prob = self._softmax_layer(cls_score, 'cls_prob')
                bbox_pred = slim.fully_connected(fc7, self._num_classes*4, weights_initializer=initializer_bbox, trainable=is_training, activation_fn=None, scope='bbox_pred')

                self._predictions['rpn_cls_score'] = rpn_cls_score
                self._predictions['rpn_cls_score_reshape'] = rpn_cls_score_reshape
                self._predictions['rpn_cls_prob'] = rpn_cls_prob
                self._predictions['rpn_bbox_pred'] = rpn_bbox_pred
                self._predictions['cls_score'] = cls_score
                self._predictions['cls_prob'] = cls_prob
                self._predictions['bbox_pred'] = bbox_pred
                self._predictions['rois'] = rois

                self._score_summaries.update(self._predictions)

            return rois, cls_prob, bbox_pred
