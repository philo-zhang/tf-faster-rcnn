from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
from datasets.imdb import imdb
import datasets.ds_utils as ds_utils
import numpy as np
import scipy.io as sio
import scipy.sparse
import utils.cython_bbox
import pickle
import subprocess
import uuid
from model.config import cfg
import pandas as pd
import json

class carback(imdb):
    def __init__(self, devkit_path=None):
        imdb.__init__(self, 'carback')
        self._devkit_path = self._get_default_path() if devkit_path is None \
                else devkit_path
        self._data_path = os.path.join(self._devkit_path)
        self._classes = ('__background__',
                        'car')
        self._class_to_ind = dict(list(zip(self.classes, list(range(self.num_classes)))))
        self._image_ext = '.jpg'
        self._image_index = self._load_image_set_index()

        self._roidb_handler = self.gt_roidb
        self._salt = str(uuid.uuid4())
        self._comp_id = 'comp4'

        self.config = {'cleanup': True,
                       'use_salt': True,
                       'use_diff': False,
                       'matlab_eval': False,
                       'rpn_file': None,
                       'min_size': 2}
        assert os.path.exists(self._devkit_path), \
                'lisa_tf path does not exist: {}'.format(self._devkit_path)
        assert os.path.exists(self._data_path), \
                'Path does not exist: {}'.format(self._data_path)

    def image_path_at(self, i):
        return os.path.join(self._data_path, self._image_index[i])

    def _get_default_path(self):
        return os.path.join(cfg.DATA_DIR, 'carback')

    def _load_image_set_index(self):
        image_index = []
        with open(os.path.join(self._data_path, 'label.json'), 'r') as f:
            data = json.load(f)
            for entry in data:
                if (len(entry['annotations']) > 0):
                    image_index.append(entry['filename'])
        return image_index

    def gt_roidb(self):
        cache_file = os.path.join(self.cache_path, self.name + '_gt_roidb.pkl')
        print (cache_file)
        if os.path.exists(cache_file):
            with open(cache_file, 'rb') as fid:
                try:
                    roidb = pickle.load(fid)
                except:
                    roidb = pickle.load(fid, encoding='bytes')
            print('{} gt roidb loaded from {}'.format(self.name, cache_file))
            return roidb
        with open(os.path.join(self._data_path, 'label.json'), 'r') as f:
            data = json.load(f)
            gt_roidb = []
            for entry in data:
                if (len(entry['annotations']) > 0) :
                    x = entry['annotations'][0]['x']
                    y = entry['annotations'][0]['y']
                    width = entry['annotations'][0]['width']
                    height = entry['annotations'][0]['height']
                    box = np.array([[x, y, x+width-1, y+height-1]])
                    overlap = np.zeros((1, self.num_classes), dtype=np.float32)
                    overlap[0,1] = 1
                    seg_area = width * height
                    overlap = scipy.sparse.csr_matrix(overlap)

                    gt_roidb.append({'boxes': box,
                                 'gt_classes': [1],
                                 'gt_overlaps': overlap,
                                 'flipped': False,
                                 'seg_areas': seg_area})

        print ('gt_roidb length: ', len(gt_roidb))

        with open(cache_file, 'wb') as fid:
            pickle.dump(gt_roidb, fid, pickle.HIGHEST_PROTOCOL)
        print ('wrote gt roidb to {}'.format(cache_file))

        return gt_roidb

if __name__ == '__main__':
     d = carback()
     res = d.roidb
