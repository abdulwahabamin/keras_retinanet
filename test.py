import keras

# import keras_retinanet
from keras_retinanet import models
from keras_retinanet.utils.image import read_image_bgr, preprocess_image, resize_image
from keras_retinanet.utils.visualization import draw_box, draw_caption
from keras_retinanet.utils.colors import label_color

# import miscellaneous modules
import matplotlib.pyplot as plt
import cv2
import os
import numpy as np
import time
from tqdm import tqdm

# set tf backend to allow memory to grow, instead of claiming everything
import tensorflow as tf

def get_session():
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    return tf.Session(config=config)

# use this environment flag to change which GPU to use
#os.environ["CUDA_VISIBLE_DEVICES"] = "1"

# set the modified tf session as backend in keras
keras.backend.tensorflow_backend.set_session(get_session())

# adjust this to point to your downloaded/trained model
# models can be downloaded here: https://github.com/fizyr/keras-retinanet/releases
# model_path = os.path.join('..', 'snapshots', 'resnet50_coco_best_v2.1.0.h5')
model_path = 'resnet50_log.h5'
# load retinanet model
model = models.load_model(model_path, backbone_name='resnet50')

# if the model is not converted to an inference model, use the line below
# see: https://github.com/fizyr/keras-retinanet#converting-a-training-model-to-inference-model
#model = models.convert_model(model)

#print(model.summary())

# load label to names mapping for visualization purposes
labels_to_names = {0: 'plastic_bag', 1: 'plastic_wrapper', 2: 'plastic_bottle', 3: 'plastic_cap', 4: 'shoes',
                   5: 'decor', 6: 'cigarette', 7: 'paper_wrapper', 8: 'cardboard', 9: 'tetrapak', 10: 'cluster',
                   11: 'other'}

# base_path = '/Ted/datasets'
# folders = ['VOC_Test_Easy','VOC_Test_Hard']
# split = 'test' #can be train, train_val or test
# savedir = '/mnt/8A2A8B2E2A8B15FB/Ted/models/results/retinanet/predict'

base_path = '/Ted/datasets/VOC_Test_'
folders = ['VOC_Test_Easy', 'VOC_Test_Hard']
split = 'test'  # can be train, train_val or test
savedir = '/Ted/results/retinanet50_log'

if not os.path.exists(savedir):
    os.mkdir(savedir)

for folder in folders:
    txt_file = os.path.join(base_path,folder,'ImageSets/Main',split + '.txt')
    f = open(txt_file,'r')
    lines = f.readlines()
    for line in tqdm(lines):
        img_name = line.strip()
        img = os.path.join(base_path,folder,'JPEGImages',img_name + '.jpg')
        # print('testing image ' + img + '\n')
        try:
            image = cv2.imread(img)
        except:
            print(img + ' does not exist')
            continue
        else:
            # copy to draw on
            draw = image.copy()
#             draw = cv2.cvtColor(draw, cv2.COLOR_BGR2RGB)

            # preprocess image for network
            image = preprocess_image(image)
            image, scale = resize_image(image)

            # process image
    #         start = time.time()
            boxes, scores, labels = model.predict_on_batch(np.expand_dims(image, axis=0))
    #         print("processing time: ", time.time() - start)

            # correct for image scale
            boxes /= scale
            annot = []
            # visualize detections
            for box, score, label in zip(boxes[0], scores[0], labels[0]):
                # scores are sorted so we can break
                if score < 0.5:
                    break

                color = label_color(label)
                # color = (0,0,255)

                b = box.astype(int)
                draw_box(draw, b, color=color)

                caption = "{} {:.2f}".format(labels_to_names[label], score)
    #             print(labels_to_names[label],score)
                annot.append(caption + ' ' + str(b[0])+ ' ' + str(b[1])+ ' ' + str(b[2])+ ' ' + str(b[3]))
            if not os.path.exists(os.path.join(savedir,folder)):
                    os.mkdir(os.path.join(savedir,folder))
            f = open(os.path.join(savedir,folder,img_name +'.txt'),'w+')
            for annotation in annot:
                f.write(annotation + '\n')
            f.close()
            draw_caption(draw, b, caption)

        cv2.imwrite(os.path.join(savedir, folder, img_name + '.jpg'), draw)
#         plt.figure(figsize=(15, 15))
#         plt.axis('off')
#         plt.imshow(draw)
#         plt.show()