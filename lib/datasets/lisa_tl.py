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

class lisa_tl(imdb):
    def __init__(self, image_set, devkit_path=None):
        imdb.__init__(self, 'lisa_tl_' + image_set)
        self._devkit_path = self._get_default_path() if devkit_path is None \
                else devkit_path
        self._data_path = os.path.join(self._devkit_path, image_set)
        self._classes = ('__background__',
                        'go', 'goForward', 'goLeft',
                        'warning', 'warningLeft',
                        'stop', 'stopLeft')
        self._class_to_ind = dict(list(zip(self.classes, list(range(self.num_classes)))))
        self._image_ext = '.png'
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
        return os.path.join(cfg.DATA_DIR, 'lisa_tl')

    def _load_image_set_index(self):
        dirs = os.listdir(self._data_path)
        image_index = []
        for dirname in dirs:
            csvfilename = os.path.join(self._data_path, dirname, 'frameAnnotationsBOX.csv')
            data = pd.read_csv(csvfilename, sep=';')
            filenames = data['Filename']
            dup = data.duplicated(subset='Filename')
            ind = [i for i, x in enumerate(dup) if not x]
            for filename in filenames.values[ind]:
                index = filename.split('/')[1]
                image_index.append(os.path.join(dirname, 'frames', index))
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
        dirs = os.listdir(self._data_path)
        data_all = pd.DataFrame([])
        for dirname in dirs:
            csvfilename = os.path.join(self._data_path, dirname, 'frameAnnotationsBOX.csv')
            data = pd.read_csv(csvfilename, sep=';')
            data_all = pd.concat([data_all, data])

        print ('image_index_length: ', len(self._image_index))
        boxes = data_all[['Upper left corner X', 'Upper left corner Y', 'Lower right corner X', 'Lower right corner Y']].values

        gt_classes = [self._class_to_ind[tag] for tag in data_all['Annotation tag'].values]

        seg_areas = (data_all['Lower right corner X'].values - data_all['Upper left corner X'].values) * (data_all['Lower right corner Y'].values - data_all['Upper left corner Y'].values)

        dup = data_all.duplicated(subset='Filename')

        inds = [i for i, x in enumerate(dup) if not x]

        gt_roidb = []
        for i in range(len(inds)):
            if i < len(inds) - 1:
                box = boxes[inds[i]:inds[i+1]]
                gt_class = gt_classes[inds[i]:inds[i+1]]
                seg_area = seg_areas[inds[i]:inds[i+1]]
            else:
                box = boxes[inds[i]:]
                gt_class = gt_classes[inds[i]:]
                seg_area = seg_areas[inds[i]:]
            overlap = np.zeros((len(gt_class), self.num_classes), dtype=np.float32)
            for j, gt in enumerate(gt_class):
                overlap[j, gt] = 1
            overlap = scipy.sparse.csr_matrix(overlap)
            gt_roidb.append({'boxes': box,
                             'gt_classes': gt_class,
                             'gt_overlaps': overlap,
                             'flipped': False,
                             'seg_areas': seg_area})
        print ('gt_roidb length: ', len(gt_roidb))

        with open(cache_file, 'wb') as fid:
            pickle.dump(gt_roidb, fid, pickle.HIGHEST_PROTOCOL)
        print ('wrote gt roidb to {}'.format(cache_file))

        return gt_roidb

if __name__ == '__main__':
     d = lisa_tl('train')
     res = d.roidb
