#---------------------
# Driva development
# squeeze net definition
#
# By Zhe zhang
#---------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import tensorflow.contrib.slim as slim
from tensorflow.contrib.slim import losses
from tensorflow.contrib.slim import arg_scope
from tensorflow.contrib.framework import add_arg_scope
import numpy as np

from nets.network import Network
from model.config import cfg

class squeeze(Network):
    def __init__(self, batch_size=1):
        Network.__init__(self, batch_size=batch_size, feat_stride=8.0)
        self._arch = 'squeeze'

    def squeeze(self, inputs, num_outputs):
        return slim.conv2d(inputs, num_outputs, [1, 1], scope='squeeze')

    def expand(self, inputs, num_outputs):
        with tf.variable_scope('expand'):
            e1x1 = slim.conv2d(inputs, num_outputs, [1, 1], scope='1x1')
            e3x3 = slim.conv2d(inputs, num_outputs, [3, 3], scope='3x3')
        return tf.concat([e1x1, e3x3], 3)

    @add_arg_scope
    def fire_module(self, inputs, squeeze_depth, expand_depth,
            reuse=None, scope=None):
        with tf.variable_scope(scope, 'fire', [inputs], reuse=reuse) as scope:
            net = self.squeeze(inputs, squeeze_depth)
            outputs = self.expand(net, expand_depth)
        return outputs

    def build_network(self, sess, is_training=True):
        with tf.variable_scope('squeeze', 'squeeze',
                regularizer=tf.contrib.layers.l2_regularizer(cfg.TRAIN.WEIGHT_DECAY)):
            if cfg.TRAIN.TRUNCATED:
                initializer = tf.truncated_normal_initializer(mean=0.0, stddev=0.01)
                initializer_bbox = tf.truncated_normal_initializer(mean=0.0, stddev=0.001)
            else:
                initializer = tf.random_normal_initializer(mean=0.0, stddev=0.01)
                initializer_bbox = tf.random_normal_initializer(mean=0.0, stddev=0.001)

            with slim.arg_scope([slim.conv2d], padding='SAME', stride=1, weights_initializer=initializer):

                net = slim.conv2d(self._image, 96, [7, 7], stride=2, scope='conv1')
                net = slim.max_pool2d(net, [3, 3], stride=2, padding='SAME', scope='maxpool1')
                net = self.fire_module(net, 16, 64, scope='fire2')
                net = self.fire_module(net, 16, 64, scope='fire3')
                net = self.fire_module(net, 32, 128, scope='fire4')
                net = slim.max_pool2d(net, [3, 3], stride=2, padding='SAME', scope='maxpool4')
                net = self.fire_module(net , 32, 128, scope='fire5')
                net = self.fire_module(net, 48, 192, scope='fire6')
                net = self.fire_module(net, 48, 192, scope='fire7')
                net = self.fire_module(net, 64, 256, scope='fire8')

                self._act_summaries.append(net)
                self._anchor_component()

                rpn = self.fire_module(net, 64, 256, scope='rpn_conv/3x3')
                self._act_summaries.append(rpn)
                rpn_cls_score = slim.conv2d(rpn, self._num_anchors * 2, [1, 1],
                        padding='VALID', activation_fn=None, scope='rpn_cls_score')
                rpn_cls_score_reshape = self._reshape_layer(rpn_cls_score, 2, 'rpn_cls_score_reshape')
                rpn_cls_prob_reshape = self._softmax_layer(rpn_cls_score_reshape, 'rpn_cls_prob_reshape')
                rpn_cls_prob = self._reshape_layer(rpn_cls_prob_reshape, self._num_anchors * 2, 'rpn_cls_prob')
                rpn_bbox_pred = slim.conv2d(rpn, self._num_anchors * 4, [1, 1], padding='VALID', activation_fn=None, scope='rpn_bbox_pred')

                if is_training:
                    rois, roi_scores = self._proposal_layer(rpn_cls_prob, rpn_bbox_pred, "rois")
                    rpn_labels = self._anchor_target_layer(rpn_cls_score, "anchor")
                    with tf.control_dependencies([rpn_labels]):
                        rois, _ = self._proposal_target_layer(rois, roi_scores, "rpn_rois")
                else:
                    if cfg.TEST.MODE == 'nms':
                        rois, _ = self._proposal_layer(rpn_cls_prob, rpn_bbox_pred, "rois")
                    elif cfg.TEST.MODE == 'top':
                        rois, _ = self._proposal_top_layer(rpn_cls_prob, rpn_bbox_pred, "rois")
                    else:
                        raise NotImplementedError

                # rcnn
                if cfg.POOLING_MODE == 'crop':
                    pool8 = self._crop_pool_layer(net, "pool8")
                else:
                    raise NotImplementedError

                fire9 = self.fire_module(pool8, 64, 256, scope='fire9')
                fire10 = self.fire_module(fire9, 96, 512, scope='fire10')
                fire11 = self.fire_module(fire10, 96, 512, scope='fire11')
                if is_training:
                    fire11 = slim.dropout(fire11, scope='drop11')
                conv12 = slim.conv2d(fire11, self._num_classes, [1, 1], scope='conv12')
                pool12 = slim.avg_pool2d(conv12, [3, 3], scope='avgpool12')
                cls_score = tf.squeeze(pool12, [1, 2], name='cls_score')
                cls_prob = self._softmax_layer(cls_score, 'cls_prob')

                conv13 = slim.conv2d(fire11, self._num_classes * 4, [1, 1], scope=
                        'conv13')
                pool13 = slim.avg_pool2d(conv13, [3, 3], scope='avgpool13')
                bbox_pred = tf.squeeze(pool13, [1, 2], name='bbox_pred')

                self._predictions["rpn_cls_score"] = rpn_cls_score
                self._predictions["rpn_cls_score_reshape"] = rpn_cls_score_reshape
                self._predictions["rpn_cls_prob"] = rpn_cls_prob
                self._predictions["rpn_bbox_pred"] = rpn_bbox_pred
                self._predictions["cls_score"] = cls_score
                self._predictions["cls_prob"] = cls_prob
                self._predictions["bbox_pred"] = bbox_pred
                self._predictions["rois"] = rois

                self._score_summaries.update(self._predictions)

                return rois, cls_prob, bbox_pred
