import numpy as np
from tensorpack import *
import scipy.io as sio
import re
import os
import cv2
import os.path
import lmdb
import argparse
np.show_config()


class RawSythtext(DataFlow):
    def __init__(self, meta, path_to_data):
        self.cellname = 'gt'
        self.textname = 'txt'
        self.imcell = 'imnames'
        self.wordname = 'wordBB'
        self.charname = 'charBB'
        self.NUMoffolder = 200
        self.imglist = meta[self.imcell][0]
        self.texts = meta[self.textname][0]
        self.wordBB = meta[self.wordname][0]
        self.charBB = meta[self.charname][0] 
        # we apply a global shuffling here because later we'll only use local shuffling
        #np.random.shuffle(self.imglist)
        self.dir = path_to_data
    def get_data(self):
        for i in range(len(self.imglist)):
            fname = os.path.join(self.dir, self.imglist[i][0])
            wordbb= self.wordBB[i]
            texts = self.texts[i]
            charbb= self.texts[i]
            with open(fname, 'rb') as f:
                jpeg = f.read()
            jpeg = np.asarray(bytearray(jpeg), dtype='uint8')
            bbox, label = _processing_image(wordbb,jpeg)
            yield [jpeg, label, bbox, texts, charbb]              
    def size(self):
        return len(self.imglist)

def _processing_image(wordbb,jpeg):
    shape = cv2.imdecode(jpeg, 1).shape
    #shape = jpeg.shape
    if(len(wordbb.shape) < 3 ):
        numofbox = 1
    else:
        numofbox = wordbb.shape[2]
    numofbox = 1 if len(wordbb.shape) < 3 else wordbb.shape[2]
    bbox = []
    [xmin, ymin]= np.min(wordbb,1)
    [xmax, ymax] = np.max(wordbb,1)
    xmin = np.maximum(xmin*1./shape[1], 0.0)
    ymin = np.maximum(ymin*1./shape[0], 0.0)
    xmax = np.minimum(xmax*1./shape[1], 1.0)
    ymax = np.minimum(ymax*1./shape[0], 1.0)
    if numofbox > 1:
        bbox = np.array([[ymin[i],xmin[i],ymax[i],xmax[i]] for i in range(numofbox)])
    if numofbox == 1:
        bbox = np.array([[ymin,xmin,ymax,xmax]])

    label = np.ones([numofbox], dtype=np.int32)
    return bbox, label

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--path_to_sythtext', help='path to sythtext data')

	args = parser.parse_args()












