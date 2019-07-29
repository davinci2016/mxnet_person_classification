from __future__ import print_function
import argparse
import numpy as np
import caffe
import os, sys
import cv2


def parse_args():
    parser = argparse.ArgumentParser(
        description='evaluate pretrained mobilenet models')
    parser.add_argument('--mode', dest='mode',
                        help="set mode", type=int)
    parser.add_argument('--proto', dest='proto',
                        help="path to deploy prototxt.", type=str)
    parser.add_argument('--model', dest='model',
                        help='path to pretrained weights', type=str)
    parser.add_argument('--image', dest='image',
                        help='path to color image', type=str)

    args = parser.parse_args()
    return args, parser


global args, parser
args, parser = parse_args()

caffe.set_mode_cpu()
net = caffe.Net(args.proto, args.model, caffe.TEST)

nh, nw = 224, 224
img_mean = np.array([101.088, 109.353, 114.978], dtype=np.float32)

resultFile = "./result.txt"
if (os.path.exists(resultFile)):
    os.remove(resultFile)

f = open(resultFile, "a+")

m = 0
def eval(img):
    if not hasattr(eval, 'n'):
        eval.n = 0
    im = caffe.io.load_image(img)
    #im = cv2.imread(img)
    #h, w, _ = im.shape
    #print (im.shape[0], im.shape[1], im.shape[2])
    h = im.shape[0]
    w = im.shape[1]

    if h < w:
        off = (w - h) / 2
        im = im[:, off:off + h]
    else:
        off = (h - w) / 2
        im = im[off:off + h, :]
    im = caffe.io.resize_image(im, [nh, nw])

    transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
    transformer.set_transpose('data', (2, 0, 1))  # row to col
    transformer.set_channel_swap('data', (2, 1, 0))  # RGB to BGR
    transformer.set_raw_scale('data', 255)  # [0,1] to [0,255]
    transformer.set_mean('data', img_mean)
    transformer.set_input_scale('data', 0.017)

    net.blobs['data'].reshape(1, 3, nh, nw)
    net.blobs['data'].data[...] = transformer.preprocess('data', im)
    out = net.forward()
    prob = out['prob']
    prob = np.squeeze(prob)
    idx = np.argsort(-prob)

    label_names = np.loadtxt('synset.txt', str, delimiter='\t')

    label = idx[0]
    print('%.2f - %s' % (prob[label], label_names[label]))
    if (label == 1):
        eval.n += 1
        item = '%d %s %s\n' % (eval.n, img, label_names[label])
        f.write(item)

print(args.image)
for dirpaths, dirnames, filenames in os.walk(args.image):
    for filename in filenames:
        origin_img_file = ('%s/%s' % (args.image, filename))
        print( '%d %s' %(m, origin_img_file))
        m += 1
        if os.path.isfile(origin_img_file):
            eval(origin_img_file)

f.close()
