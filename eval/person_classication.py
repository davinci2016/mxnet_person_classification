import numpy as np  
import sys,os  
import cv2
import time
#caffe_root = '/home/davinci/src/dnn/caffe-mobilenetv3/caffe/'
#sys.path.insert(0, caffe_root + 'python')  
import caffe  

model_root='/home/davinci/src/dnn/caffe-mobilenetv3/caffe/models/mobilenetv2'


net_file= 'mobilenet_v2_deploy.prototxt'
caffe_model='caffemodel/caffe_mobilenetv2_train_iter_60000.caffemodel'
test_dir = "/home/davinci/dnn/my_person/baidu/person-cn"

if not os.path.exists(caffe_model):
    print("MobileNetSSD_deploy.caffemodel does not exist,")
    print("use merge_bn.py to generate it.")
    exit()

caffe.set_mode_gpu()
caffe.set_device(0)
net = caffe.Net(net_file,caffe_model,caffe.TEST)  

CLASSES = ( "person", 'background')

def classication(im):
    ##  origimg = cv2.imread(imgfile)
    frame = im
    im = im / 255.0 ##normalization [0.0, 1.0]
    nh, nw = 224, 224
    img_mean = np.array([101.088, 109.353, 114.978], dtype=np.float32)

    #im = caffe.io.load_image(args.image)
    h, w, _ = im.shape

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

    start=time.time()
    out = net.forward()
    use_time=time.time()-start
    print ("time="+str(use_time)+"s") 
    fps=1/use_time
    print ("FPS="+str(fps)) 

    out = net.forward()
    prob = out['prob']
    prob = np.squeeze(prob)
    idx = np.argsort(-prob)

    label = idx[0]
    print('%.2f - %s' % (prob[label], CLASSES[label]))
    title = "%s:%.2f" % (CLASSES[label], prob[label])
    p3 = (30,30)
    cv2.putText(frame, title, p3, cv2.FONT_ITALIC, 0.6, (0, 255, 0), 1)
    cv2.imshow("MobileNetV2", frame)

cap = cv2.VideoCapture(0)
while(1):
    ret, frame = cap.read()
    classication(frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
       break

cap.release()
cv2.destroyAllWindows() 
