#!/usr/bin/env python

# --------------------------------------------------------
# Faster R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# Modified by Gen Li
# --------------------------------------------------------

"""
Demo script showing text detections in sample images.

See README.md for installation instructions before running.
"""
import matplotlib
matplotlib.use('pdf')
import _init_paths
from fast_rcnn.config import cfg
from fast_rcnn.test import im_detect
from fast_rcnn.nms_wrapper import nms
from utils.timer import Timer
import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio
import caffe, os, sys, cv2
import argparse
import re
import lxml.html as lh
import requests
import json
import time

from os import listdir
from os.path import isfile, join

CLASSES = ('__background__', 'text')

NETS = {'vgg16': ('VGG16', 'vgg16_faster_rcnn_coco_text.caffemodel'),
        'zf': ('ZF', 'ZF_faster_rcnn_final.caffemodel')}

def vis_detections(ax, class_name, dets, thresh=0.5):
  """Draw detected bounding boxes."""
  inds = np.where(dets[:, -1] >= thresh)[0]
  if len(inds) == 0:
    return False

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
                          thresh), fontsize=14)
  plt.axis('off')
  plt.tight_layout()
  plt.draw()
  return True

def im_det(net, im, imname):
  # Detect all object classes and regress object bounds
  timer = Timer()
  timer.tic()
  scores, boxes = im_detect(net, im)
  timer.toc()
  print ('Detection took {:.3f}s for '
       '{:d} object proposals').format(timer.total_time, boxes.shape[0])

  # Visualize detections for each class
  CONF_THRESH = 0.8
  NMS_THRESH = 0.3
  im = im[:, :, (2, 1, 0)]
  fig, ax = plt.subplots(figsize=(12, 12))
  ax. imshow(im, aspect='equal')
  detected = False
  for cls_ind, cls in enumerate(CLASSES[1:]):
    cls_ind += 1 # because we skipped background
    cls_boxes = boxes[:, 4*cls_ind:4*(cls_ind + 1)]
    cls_scores = scores[:, cls_ind]
    dets = np.hstack((cls_boxes,
              cls_scores[:, np.newaxis])).astype(np.float32)
    keep = nms(dets, NMS_THRESH)
    dets = dets[keep, :]
    detected = vis_detections(ax, cls, dets, thresh=CONF_THRESH)
  if detected:
    detect_output_dir = os.path.join(cfg.ROOT_DIR, 'output_img',
    str(imname) + "_detect_rst.jpg")
    plt.savefig(detect_output_dir)
    ori_output_dir = os.path.join(cfg.ROOT_DIR, 'output_img',
    str(imname) + ".jpg")
    cv2.imwrite(ori_output_dir, im)

def parse_args():
  """Parse input arguments."""
  parser = argparse.ArgumentParser(description='Faster R-CNN demo')
  parser.add_argument('--gpu', dest='gpu_id', help='GPU device id to use [0]',
            default=0, type=int)
  parser.add_argument('--cpu', dest='cpu_mode',
            help='Use CPU mode (overrides --gpu)',
            action='store_true')
  parser.add_argument('--net', dest='demo_net', help='Network to use [vgg16]',
            choices=NETS.keys(), default='vgg16')
  parser.add_argument('--model', dest='trained_model',
            help='weights')
  default_dir = os.path.join(cfg.ROOT_DIR, '..', 'datasets', 'visual-test')
  parser.add_argument('--video_url', dest='dataset_dir',
            help='dataset_dir',
            default=default_dir)
  args = parser.parse_args()
  return args

def caffe_init(args):
  cfg.TEST.HAS_RPN = True  # Use RPN for proposals
  prototxt = os.path.join(cfg.ROOT_DIR, 'models', 'coco_text',
            NETS[args.demo_net][0], 'faster_rcnn_end2end', 'test.prototxt')
  if args.trained_model:
    caffemodel = os.path.join(cfg.DATA_DIR, 'faster_rcnn_models',
                args.trained_model)
  else:
    caffemodel = os.path.join(cfg.DATA_DIR, 'faster_rcnn_models',
                NETS[args.demo_net][1])

  if not os.path.isfile(caffemodel):
    raise IOError(('{:s} not found.\nDid you run ./data/script/'
             'fetch_faster_rcnn_models.sh?').format(caffemodel))

  if args.cpu_mode:
    caffe.set_mode_cpu()
  else:
    caffe.set_mode_gpu()
    caffe.set_device(args.gpu_id)
    cfg.GPU_ID = args.gpu_id
  net = caffe.Net(prototxt, caffemodel, caffe.TEST)
  print '\n\nLoaded network {:s}'.format(caffemodel)
  # Warmup on a dummy image
  im = 128 * np.ones((300, 500, 3), dtype=np.uint8)
  for i in xrange(2):
    _, _= im_detect(net, im)
  return net

def url_convert(url):
  r = requests.get(url)
  doc = lh.fromstring(r.content)
  gid = doc.xpath('//@tt-videoid')
  json_url = 'http://ii.snssdk.com/video/urls/1/toutiao/mp4/' + str(gid[0]) + '?nobase64=true'
  r = requests.get(json_url)
  dic = json.loads(r.content)
  if "data" in dic:
    if "video_list" in dic["data"]:
      if "video_1" in dic["data"]["video_list"]:
        if "main_url" in dic["data"]["video_list"]["video_1"]:
          return str(dic["data"]["video_list"]["video_1"]["main_url"]), gid[0]
  return 'bad url', gid[0]

if __name__ == '__main__':
  args = parse_args()
  net = caffe_init(args)
  # url = args.video_url
  url = 'http://toutiao.com/a6308112164441161986/'
  real_url, gid = url_convert(url)
  print real_url
  os.system('curl -o test.mp4 ' + real_url)
  capture = cv2.VideoCapture('./test.mp4')
  size = (int(capture.get(cv2.cv.CV_CAP_PROP_FRAME_WIDTH)),
          int(capture.get(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT)))
  fps = capture.get(cv2.cv.CV_CAP_PROP_FPS)
  print fps
  cnt = 0
  time_start = time.time()
  timer = Timer()
  timer.tic()
  while True:
    ret, img = capture.read()
    if (type(img) == type(None)):
      break
    if (0xFF & cv2.waitKey(5) == 27) or img.size == 0:
      break
    if cnt % fps == 0:
      print str(cnt / fps) + ', seconds'
      img = cv2.resize(img, (img.shape[1]/2, img.shape[0]/2))
      seconds = cnt / fps;
      imname = str(gid) + '_' + str(seconds) + 's'
      im_det(net, img, imname)
    cnt += 1
  timer.toc()
  print ('Detection took {:.3f}s for '
       'video {}').format(timer.total_time, real_url)
  capture.release()
  cv2.destroyAllWindows()
