#!/usr/bin/env python

# --------------------------------------------------------
# Tensorflow Faster R-CNN
# Licensed under The MIT License [see LICENSE for details]
# Written by Xinlei Chen, based on code from Ross Girshick
# --------------------------------------------------------

"""
Demo script showing detections in sample images.

See README.md for installation instructions before running.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import _init_paths
from model.config import cfg
#from model.test import im_detect
from model.test import _get_blobs
from model.bbox_transform import bbox_transform_inv, clip_boxes
from model.nms_wrapper import nms
from layer_utils.generate_anchors import generate_anchors

from utils.timer import Timer
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import os, cv2
import argparse

from nets.vgg16 import vgg16
from nets.inception_v1 import inception_v1

CLASSES = ('__background__',
           'go', 'goForward', 'goLeft',
           'warning', 'warningLeft',
           'stop', 'stopLeft')

NETS = {'inception_v1': ('inception_v1_faster_rcnn_iter_70000.ckpt', 'inception_v1.weights')}

def vis_detections(im, class_name, dets, thresh=0.5):
    """Draw detected bounding boxes."""
    inds = np.where(dets[:, -1] >= thresh)[0]
    if len(inds) == 0:
        return

    #print (dets[inds])

    im = im[:, :, (2, 1, 0)]
    fig, ax = plt.subplots(figsize=(12, 12))
    ax.imshow(im, aspect='equal')
    for i in inds:
        bbox = dets[i, :4]
        score = dets[i, -1]

        ax.add_patch(
            plt.Rectangle((bbox[0], bbox[1]),
                          bbox[2] - bbox[0],
                          bbox[3] - bbox[1], fill=False,
                          edgecolor='red', linewidth=3.5)
            )
        ax.text(bbox[0], bbox[1] - 2,
                '{:s} {:.3f}'.format(class_name, score),
                bbox=dict(facecolor='blue', alpha=0.5),
                fontsize=14, color='white')

    ax.set_title(('{} detections with '
                  'p({} | box) >= {:.1f}').format(class_name, class_name,
                                                  thresh),
                  fontsize=14)
    plt.axis('off')
    plt.tight_layout()
    plt.draw()

def generate_anchors_pre(im_info, feat_stride, anchor_scales):
    height = int(np.ceil(im_info[0,0] / feat_stride))
    width = int(np.ceil(im_info[0,1] / feat_stride))
    anchors = generate_anchors(scales=np.array(anchor_scales))
    A = anchors.shape[0]
    #print ('width, feat_stride: ', width, feat_stride)
    shift_x = np.arange(0, width) * feat_stride
    shift_y = np.arange(0, height) * feat_stride
    shift_x, shift_y = np.meshgrid(shift_x, shift_y)
    shifts = np.vstack((shift_x.ravel(), shift_y.ravel(), shift_x.ravel(), shift_y.ravel())).transpose()
    K = shifts.shape[0]
    # width changes faster, so here it is H, W, C
    anchors = anchors.reshape((1, A, 4)) + shifts.reshape((1, K, 4)).transpose((1, 0, 2))
    anchors = anchors.reshape((K * A, 4)).astype(np.float32, copy=False)
    length = np.int32(anchors.shape[0])
    return anchors

def proposal_layer(rpn_cls_prob, rpn_bbox_pred, im_info, cfg_key, feat_stride, anchors, anchor_scales):
  if type(cfg_key) == bytes:
      cfg_key = cfg_key.decode('utf-8')
  pre_nms_topN = cfg[cfg_key].RPN_PRE_NMS_TOP_N
  post_nms_topN = cfg[cfg_key].RPN_POST_NMS_TOP_N
  nms_thresh = cfg[cfg_key].RPN_NMS_THRESH

  scales = np.array(anchor_scales)
  num_anchors = scales.shape[0] * 3
  im_info = im_info[0]
  # Get the scores and bounding boxes
  scores = rpn_cls_prob[:, :, :, num_anchors:]
  rpn_bbox_pred = rpn_bbox_pred.reshape((-1, 4))
  scores = scores.reshape((-1, 1))
  proposals = bbox_transform_inv(anchors, rpn_bbox_pred)
  proposals = clip_boxes(proposals, im_info[:2])

  print ('proposals: ', proposals[-18:,:])

  # Pick the top region proposals
  order = scores.ravel().argsort()[::-1]
  if pre_nms_topN > 0:
    order = order[:pre_nms_topN]
  proposals = proposals[order, :]
  scores = scores[order]

  print ('orders: ', order[0:9])

  print ('top proposals pre: ', proposals[0:9,:])
  # Non-maximal suppression
  keep = nms(np.hstack((proposals, scores)), nms_thresh)

  # Pick th top region proposals after NMS
  if post_nms_topN > 0:
    keep = keep[:post_nms_topN]
  proposals = proposals[keep, :]
  scores = scores[keep]

  print ('top proposals: ', proposals[0:9,:])

  # Only support single image as input
  batch_inds = np.zeros((proposals.shape[0], 1), dtype=np.float32)
  blob = np.hstack((batch_inds, proposals.astype(np.float32, copy=False)))

  return blob, scores

def im_detect(sess, im):
  blobs, im_scales = _get_blobs(im)
  assert len(im_scales) == 1, "Only single-image batch implemented"

  im_blob = blobs['data']
  # seems to have height, width, and image scales
  # still not sure about the scale, maybe full image it is 1.
  blobs['im_info'] = np.array([[im_blob.shape[1], im_blob.shape[2], im_scales[0]]], dtype=np.float32)
  feat_stride = 4.0
  anchor_scales = [2,4,8]


  rpn_cls_prob_tensor = sess.graph.get_tensor_by_name('InceptionV1/rpn_cls_prob/transpose_1:0')
  rpn_bbox_pred_tensor = sess.graph.get_tensor_by_name('InceptionV1/rpn_bbox_pred/BiasAdd:0')
  cls_prob_tensor = sess.graph.get_tensor_by_name('InceptionV1/cls_prob:0')
  bbox_pred_tensor = sess.graph.get_tensor_by_name('InceptionV1/bbox_pred/BiasAdd:0')
  fc6_tensor = sess.graph.get_tensor_by_name('InceptionV1/fc6/Relu:0')
  fc7_tensor = sess.graph.get_tensor_by_name('InceptionV1/fc7/Relu:0')
  matmul_tensor = sess.graph.get_tensor_by_name('InceptionV1/bbox_pred/MatMul:0')
  #cv2.imshow('image', np.squeeze(im_blob))
  #cv2.waitKey(0)
  rpn_cls_prob, rpn_bbox_pred = sess.run([rpn_cls_prob_tensor,
                                      rpn_bbox_pred_tensor],
                                      feed_dict={'Placeholder:0': blobs['data']})


  anchors = generate_anchors_pre(blobs['im_info'], feat_stride, anchor_scales)

  print ('anchors: ', anchors[-18:,:])
  #print ('anchor shape: ', anchors.shape)

  rois,_ = proposal_layer(rpn_cls_prob, rpn_bbox_pred, blobs['im_info'], 'TEST', feat_stride, anchors, anchor_scales)
  print ('rois.shape: ', rois.shape)

  scores, bbox_pred, fc6, fc7, matmul = sess.run([cls_prob_tensor,
                                bbox_pred_tensor,
                                fc6_tensor,
                                fc7_tensor,
                                matmul_tensor],
                                feed_dict={'Placeholder:0': blobs['data'], 'Placeholder_3:0': 1, 'Placeholder_4:0': rois})
  #bbox_pred[:,0:2] = bbox_pred[:,0:2]/10
  #bbox_pred[:,2:4] = bbox_pred[:,2:4]/20
  #print ('fc6: ', fc6[0:5,0:5])
  #print ('fc7: ', fc7[0:5,0:5])

  #print ('scores: ', scores[0:10,0:2])
  #print ('rois: ', rois[0:10])
  #print ('bbox_pred: ', bbox_pred[0:10,0:2])
  #print ('matmul: ', matmul[0:5,0:5])

  weights_tensor = sess.graph.get_tensor_by_name('InceptionV1/bbox_pred/weights:0')
  weights = sess.run(weights_tensor)
  #print ('weights: ', weights[0:5,0:5])

  boxes = rois[:, 1:5] / im_scales[0]
  #print ('rois and boxes: ', rois, boxes)
  #print(scores.shape, bbox_pred.shape, rois.shape, boxes.shape)
  #print ('bbox_pred: ', bbox_pred[0])
  cfg.TEST.BBOX_REG = False
  if cfg.TEST.BBOX_REG:
    # Apply bounding-box regression deltas
    box_deltas = bbox_pred
    pred_boxes = bbox_transform_inv(boxes, box_deltas)
    pred_boxes = _clip_boxes(pred_boxes, im.shape)
  else:
    # Simply repeat the boxes, once for each class
    pred_boxes = np.tile(boxes, (1, scores.shape[1]))

  return scores, pred_boxes

def _clip_boxes(boxes, im_shape):
  """Clip boxes to image boundaries."""
  # x1 >= 0
  boxes[:, 0::4] = np.maximum(boxes[:, 0::4], 0)
  # y1 >= 0
  boxes[:, 1::4] = np.maximum(boxes[:, 1::4], 0)
  # x2 < im_shape[1]
  boxes[:, 2::4] = np.minimum(boxes[:, 2::4], im_shape[1] - 1)
  # y2 < im_shape[0]
  boxes[:, 3::4] = np.minimum(boxes[:, 3::4], im_shape[0] - 1)
  return boxes


def demo(sess, net, image_name):
    """Detect object classes in an image using pre-computed object proposals."""

    # Load the demo image
    im_file = os.path.join('data/lisa_tl', image_name)
    im = cv2.imread(im_file)

    # Detect all object classes and regress object bounds
    timer = Timer()
    timer.tic()
    scores, boxes = im_detect(sess, im)
    timer.toc()
    print('Detection took {:.3f}s for {:d} object proposals'.format(timer.total_time, boxes.shape[0]))

    # Visualize detections for each class
    CONF_THRESH = 0.9
    NMS_THRESH = 0.3
    for cls_ind, cls in enumerate(CLASSES[1:]):
        cls_ind += 1 # because we skipped background
        cls_boxes = boxes[:, 4*cls_ind:4*(cls_ind + 1)]
        cls_scores = scores[:, cls_ind]
        dets = np.hstack((cls_boxes,
                          cls_scores[:, np.newaxis])).astype(np.float32)
        keep = nms(dets, NMS_THRESH)
        dets = dets[keep, :]
        #print (dets)
        vis_detections(im, cls, dets, thresh=CONF_THRESH)

def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='Tensorflow Faster R-CNN demo')
    parser.add_argument('--net', dest='demo_net', help='Network to use [vgg16]',
                        choices=NETS.keys(), default='inception_v1')

    args = parser.parse_args()

    return args

if __name__ == '__main__':
    cfg.TEST.HAS_RPN = True  # Use RPN for proposals
    cfg.POOLING_SIZE = 2
    args = parse_args()

    # model path
    #demonet = args.demo_net
    #tfmodel = os.path.join('output', 'inception_v1', 'lisa_tl_train', 'default',
    #                          NETS[demonet][0])
    #if not os.path.isfile(tfmodel + '.meta'):
    #    raise IOError(('{:s} not found.\nDid you run ./data/script/'
    #                   'fetch_faster_rcnn_models.sh?').format(tfmodel + '.meta'))

    # set config
    tfconfig = tf.ConfigProto(allow_soft_placement=True)
    tfconfig.gpu_options.allow_growth=True

    # init session
    sess = tf.Session(config=tfconfig)
    # load network
    if demonet == 'vgg16':
        net = vgg16(batch_size=1)
    elif demonet == 'inception_v1':
        net = inception_v1(batch_size=1)
    else:
        raise NotImplementedError

    #net.create_architecture(sess, "TEST", 8,
    #                      tag='default', anchor_scales=[1, 2, 3])
    #print (tfmodel)
    #saver = tf.train.Saver()
    #saver.restore(sess, tfmodel)

    graph_def = tf.GraphDef()
    with tf.gfile.FastGFile('output/inception_v1/lisa_tl_train/default/output_graph.pb', 'rb') as f:
        graph_def.ParseFromString(f.read())
        tf.import_graph_def(graph_def, name='')

    print (sess.graph.get_operations())
    #output_graph_def = tf.graph_util.convert_variables_to_constants(
    #        sess, sess.graph.as_graph_def(), ['InceptionV1/cls_prob', 'InceptionV1/bbox_pred/BiasAdd'])
    #with tf.gfile.FastGFile('output_graph.pb', 'wb') as f:
    #    f.write(output_graph_def.SerializeToString())


    print('Loaded network {:s}'.format(tfmodel))

    im_names = ['0005.png', '0006.png', '0007.png']
    for im_name in im_names:
        print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
        print('Demo for data/demo/{}'.format(im_name))
        demo(sess, net, im_name)

    plt.show()
