from __future__ import print_function
import argparse
import numpy as np
import os, sys
import cv2
import time

from mxnet import nd, image
from gluoncv.model_zoo import get_model
import mxnet as mx

from mxnet.gluon.data.vision import transforms

parser = argparse.ArgumentParser(description='Predict ImageNet classes from a given image')
parser.add_argument('--model', type=str, required=True,
                    help='name of the model to use')
parser.add_argument('--load-params', type=str, default='', required=True,
                    help='path to the saved model parameters')
parser.add_argument('--input-pic', type=str, default='',
                    help='path to the input picture')
parser.add_argument('--image-dir', type=str, default='',
                    help='path to color image')
opt = parser.parse_args()

# Load Model
num_gpus = 1
classes = ['Person', 'BackGound']

context = [mx.gpu(i) for i in range(num_gpus)] if num_gpus > 0 else [mx.cpu()]
model_name = opt.model
pretrained = True if opt.load_params == '' else False

net = get_model(model_name, pretrained=pretrained, classes=2, ctx=context)
net.load_parameters(opt.load_params)
net.hybridize(static_alloc=True, static_shape=True)

resultFile = "./result.txt"
if os.path.exists(resultFile):
    os.remove(resultFile)

f = open(resultFile, "a+")


def transform_eval(imgs, resize_short=256, crop_size=224,
                   mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
    """A util function to transform all images to tensors as network input by applying
    normalizations. This function support 1 NDArray or iterable of NDArrays.

    Parameters
    ----------
    imgs : NDArray or iterable of NDArray
        Image(s) to be transformed.
    resize_short : int, default=256
        Resize image short side to this value and keep aspect ratio.
    crop_size : int, default=224
        After resize, crop the center square of size `crop_size`
    mean : iterable of float
        Mean pixel values.
    std : iterable of float
        Standard deviations of pixel values.

    Returns
    -------
    mxnet.NDArray or list of such tuple
        A (1, 3, H, W) mxnet NDArray as input to network
        If multiple image names are supplied, return a list.
    """
    if isinstance(imgs, mx.nd.NDArray):
        imgs = [imgs]
    for im in imgs:
        assert isinstance(im, mx.nd.NDArray), "Expect NDArray, got {}".format(type(im))

    transform_fn = transforms.Compose([
        transforms.Resize(resize_short, keep_ratio=True),
        transforms.CenterCrop(crop_size),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])

    res = [transform_fn(img).expand_dims(0) for img in imgs]

    if len(res) == 1:
        return res[0]
    return res


def eval(img):
    # Transform
    img = transform_eval(img)
    pred = net(img)

    ind = nd.topk(pred, k=1)[0].astype('int')
    #print('The input picture is classified to be')
    label = ind[0].asscalar()
    prob = nd.softmax(pred)[0][label].asscalar()
    return label, prob

def eval_pic(pic):
    # Load Images
    img = image.imread(pic)
    label, prob = eval(img)
    print('\t[%s], with probability %.3f.' % (classes[label], prob))

def classication():
    cap = cv2.VideoCapture(0)
    while 1:
        ret, frame = cap.read()

        start = time.time()
        img = nd.array(frame)
        label, prob = eval(img)
        use_time = time.time() - start
        print("time=" + str(use_time) + "s")
        fps = 1 / use_time
        print("FPS=" + str(fps))

        print('%.2f - %s' % (prob, classes[label]))
        title = "%s:%.2f" % (classes[label], prob)
        p3 = (30, 30)
        if label == 0:
            cv2.putText(frame, title, p3, cv2.FONT_ITALIC, 0.6, (0, 0, 255), 2)
        else:
            cv2.putText(frame, title, p3, cv2.FONT_ITALIC, 0.6, (0, 255, 0), 2)
        cv2.imshow(model_name, frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


m = 0
if opt.image_dir != '':
    print(opt.image_dir)
    for dirpaths, dirnames, filenames in os.walk(opt.image_dir):
        for filename in filenames:
            file = ('%s/%s' % (opt.image_dir, filename))
            print('%d %s' % (m, file))
            m += 1
            if os.path.isfile(file):
                eval_pic(file)
elif opt.input_pic != '':
    eval_pic(opt.input_pic)
else:
    classication()

f.close()
