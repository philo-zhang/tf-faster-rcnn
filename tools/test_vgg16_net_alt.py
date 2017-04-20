import _init_paths
#from model.test_vgg16 import test_net
from model.config import cfg, cfg_from_file, cfg_from_list, get_output_dir
from datasets.factory import get_imdb
import argparse
import pprint
import time, os, sys

import tensorflow as tf
from nets.vgg16_depre import vgg16

from utils.cython_nms import nms
from model.bbox_transform import clip_boxes, bbox_transform_inv
from utils.blob import im_list_to_blob
import numpy as np
from utils.timer import Timer
import cv2

def parse_args():
  """
  Parse input arguments
  """
  parser = argparse.ArgumentParser(description='Test a Fast R-CNN network')
  parser.add_argument('--cfg', dest='cfg_file',
            help='optional config file', default=None, type=str)
  parser.add_argument('--model', dest='model',
            help='model to test',
            default=None, type=str)
  parser.add_argument('--weight', dest='weight',
            help='weight to test',
            default=None, type=str)
  parser.add_argument('--imdb', dest='imdb_name',
            help='dataset to test',
            default='voc_2007_test', type=str)
  parser.add_argument('--comp', dest='comp_mode', help='competition mode',
            action='store_true')
  parser.add_argument('--num_dets', dest='max_per_image',
            help='max number of detections per image',
            default=100, type=int)
  parser.add_argument('--tag', dest='tag',
                        help='tag of the model',
                        default='', type=str)
  parser.add_argument('--set', dest='set_cfgs',
                        help='set config keys', default=None,
                        nargs=argparse.REMAINDER)

  if len(sys.argv) == 1:
    parser.print_help()
    sys.exit(1)

  args = parser.parse_args()
  return args

def _get_image_blob(im):
  """Converts an image into a network input.
  Arguments:
    im (ndarray): a color image in BGR order
  Returns:
    blob (ndarray): a data blob holding an image pyramid
    im_scale_factors (list): list of image scales (relative to im) used
      in the image pyramid
  """
  im_orig = im.astype(np.float32, copy=True)
  im_orig -= cfg.PIXEL_MEANS

  im_shape = im_orig.shape
  im_size_min = np.min(im_shape[0:2])
  im_size_max = np.max(im_shape[0:2])

  processed_ims = []
  im_scale_factors = []

  for target_size in cfg.TEST.SCALES:
    im_scale = float(target_size) / float(im_size_min)
    # Prevent the biggest axis from being more than MAX_SIZE
    if np.round(im_scale * im_size_max) > cfg.TEST.MAX_SIZE:
      im_scale = float(cfg.TEST.MAX_SIZE) / float(im_size_max)
    im = cv2.resize(im_orig, None, None, fx=im_scale, fy=im_scale,
            interpolation=cv2.INTER_LINEAR)
    im_scale_factors.append(im_scale)
    processed_ims.append(im)

  # Create a blob to hold the input images
  blob = im_list_to_blob(processed_ims)

  return blob, np.array(im_scale_factors)

def _get_blobs(im):
  """Convert an image and RoIs within that image into network inputs."""
  blobs = {}
  blobs['data'], im_scale_factors = _get_image_blob(im)

  return blobs, im_scale_factors

def test_image(sess, image, im_info):
    feed_dict = {'Placeholder:0': image,
            'Placeholder_1:0': im_info}
    tensor_cls_score = sess.graph.get_tensor_by_name("vgg16_default/cls_score/BiasAdd:0")
    tensor_cls_prob = sess.graph.get_tensor_by_name("vgg16_default/cls_prob:0")
    tensor_bbox_pred = sess.graph.get_tensor_by_name("vgg16_default/bbox_pred/BiasAdd:0")
    tensor_rois = sess.graph.get_tensor_by_name("vgg16_default/rois/PyFunc:0")
    tensor_conv5_3 = sess.graph.get_tensor_by_name("vgg16_default/conv5_3/Relu:0")
    rois = sess.run([tensor_conv5_3], feed_dict=feed_dict)

    feed_dict = {'vgg16_default/rpn_rois/to_int32:0': rois}

    #cls_score, cls_prob, bbox_pred = sess.run([tensor_cls_score,
    #                                           tensor_cls_prob,
    #                                           tensor_bbox_pred],
    #                                           feed_dict=feed_dict)

    #cls_score, cls_prob, bbox_pred, rois = sess.run([tensor_cls_score,
    #                                                 tensor_cls_prob,
    #                                                 tensor_bbox_pred,
    #                                                 tensor_rois],
    #                                                feed_dict=feed_dict)
    return cls_score, cls_prob, bbox_pred, rois


def im_detect(sess, im):
  blobs, im_scales = _get_blobs(im)
  assert len(im_scales) == 1, "Only single-image batch implemented"

  im_blob = blobs['data']
  # seems to have height, width, and image scales
  # still not sure about the scale, maybe full image it is 1.
  blobs['im_info'] = np.array([[im_blob.shape[1], im_blob.shape[2], im_scales[0]]], dtype=np.float32)

  _, scores, bbox_pred, rois = test_image(sess, blobs['data'], blobs['im_info'])

  boxes = rois[:, 1:5] / im_scales[0]

  if cfg.TEST.BBOX_REG:
    # Apply bounding-box regression deltas
    box_deltas = bbox_pred
    pred_boxes = bbox_transform_inv(boxes, box_deltas)
    pred_boxes = _clip_boxes(pred_boxes, im.shape)
  else:
    # Simply repeat the boxes, once for each class
    pred_boxes = np.tile(boxes, (1, scores.shape[1]))

  return scores, pred_boxes

def test_net(sess, imdb, weights_filename, max_per_image=100, thresh=0.05):
  np.random.seed(cfg.RNG_SEED)
  """Test a Fast R-CNN network on an image database."""
  num_images = len(imdb.image_index)
  # all detections are collected into:
  #  all_boxes[cls][image] = N x 5 array of detections in
  #  (x1, y1, x2, y2, score)
  all_boxes = [[[] for _ in xrange(num_images)]
         for _ in xrange(imdb.num_classes)]

  output_dir = get_output_dir(imdb, weights_filename)
  # timers
  _t = {'im_detect' : Timer(), 'misc' : Timer()}

  for i in xrange(num_images):
    im = cv2.imread(imdb.image_path_at(i))

    _t['im_detect'].tic()
    scores, boxes = im_detect(sess, im)
    _t['im_detect'].toc()

    _t['misc'].tic()

    # skip j = 0, because it's the background class
    for j in xrange(1, imdb.num_classes):
      inds = np.where(scores[:, j] > thresh)[0]
      cls_scores = scores[inds, j]
      cls_boxes = boxes[inds, j*4:(j+1)*4]
      cls_dets = np.hstack((cls_boxes, cls_scores[:, np.newaxis])) \
        .astype(np.float32, copy=False)
      keep = nms(cls_dets, cfg.TEST.NMS)
      cls_dets = cls_dets[keep, :]
      all_boxes[j][i] = cls_dets

    # Limit to max_per_image detections *over all classes*
    if max_per_image > 0:
      image_scores = np.hstack([all_boxes[j][i][:, -1]
                    for j in xrange(1, imdb.num_classes)])
      if len(image_scores) > max_per_image:
        image_thresh = np.sort(image_scores)[-max_per_image]
        for j in xrange(1, imdb.num_classes):
          keep = np.where(all_boxes[j][i][:, -1] >= image_thresh)[0]
          all_boxes[j][i] = all_boxes[j][i][keep, :]
    _t['misc'].toc()

    print 'im_detect: {:d}/{:d} {:.3f}s {:.3f}s' \
        .format(i + 1, num_images, _t['im_detect'].average_time,
            _t['misc'].average_time)

    for m in range(1, imdb.num_classes):
        tmp_boxes = all_boxes[m][i]
        for n in range(tmp_boxes.shape[0]):
            tmp_box = tmp_boxes[n]
            cv2.rectangle(im, (tmp_box[0], tmp_box[1]), (tmp_box[2], tmp_box[3]), (255,0,0))
    cv2.imshow('image', im)
    cv2.waitKey(1)


  det_file = os.path.join(output_dir, 'detections.pkl')
  with open(det_file, 'wb') as f:
    cPickle.dump(all_boxes, f, cPickle.HIGHEST_PROTOCOL)

  print 'Evaluating detections'
  imdb.evaluate_detections(all_boxes, output_dir)


if __name__ == '__main__':
  args = parse_args()

  print('Called with args:')
  print(args)

  if args.cfg_file is not None:
    cfg_from_file(args.cfg_file)
  if args.set_cfgs is not None:
    cfg_from_list(args.set_cfgs)

  print('Using config:')
  pprint.pprint(cfg)

  # if has model, get the name from it
  # if does not, then just use the inialization weights
  filename = os.path.splitext(os.path.basename(args.model))[0]

  tag = args.tag
  tag = tag if tag else 'default'
  filename = tag + '/' + filename

  imdb = get_imdb(args.imdb_name)
  imdb.competition_mode(args.comp_mode)

  tfconfig = tf.ConfigProto(allow_soft_placement=True)
  tfconfig.gpu_options.allow_growth=True

  # init session
  sess = tf.Session(config=tfconfig)
  # load network
  #net = vgg16(batch_size=1)
  # load model
  if imdb.name.startswith('voc'):
    anchors = [8, 16, 32]
  else:
    anchors = [4, 8, 16, 32]

  #net.create_architecture(sess, "TEST", imdb.num_classes, caffe_weight_path=args.weight,
  #                        tag='default', anchor_scales=anchors)

  print ('Loading model check point from {:s}').format(args.model)
  saver = tf.train.import_meta_graph(args.model + '.meta')
  saver.restore(sess, args.model)
  #print (sess.graph.get_operations())
  print 'Loaded.'

  test_net(sess, imdb, filename, max_per_image=args.max_per_image)

  sess.close()
