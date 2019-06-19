from __future__ import division
import os
import cv2
import numpy as np
import pickle
import time
from keras_frcnn import config
from keras import backend as K
from keras.layers import Input
from keras.models import Model
from keras_frcnn import roi_helpers
import argparse
import os
import keras_frcnn.resnet as nn
from keras_frcnn.visualize import draw_boxes_and_label_on_image_cv2
import matplotlib
matplotlib.use('Agg')


def format_img_size(img, cfg):
    """ formats the image size based on config """
    img_min_side = float(cfg.im_size)
    (height, width, _) = img.shape

    if width <= height:
        ratio = img_min_side / width
        new_height = int(ratio * height)
        new_width = int(img_min_side)
    else:
        ratio = img_min_side / height
        new_width = int(ratio * width)
        new_height = int(img_min_side)
    img = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_CUBIC)
    return img, ratio


def format_img_channels(img, cfg):
    """ formats the image channels based on config """
    img = img[:, :, (2, 1, 0)]
    img = img.astype(np.float32)
    img[:, :, 0] -= cfg.img_channel_mean[0]
    img[:, :, 1] -= cfg.img_channel_mean[1]
    img[:, :, 2] -= cfg.img_channel_mean[2]
    img /= cfg.img_scaling_factor
    img = np.transpose(img, (2, 0, 1))
    img = np.expand_dims(img, axis=0)
    return img


def format_img(img, C):
    """ formats an image for model prediction based on config """
    img, ratio = format_img_size(img, C)
    img = format_img_channels(img, C)
    return img, ratio


# Method to transform the coordinates of the bounding box to its original size
def get_real_coordinates(ratio, x1, y1, x2, y2):
    real_x1 = int(round(x1 // ratio))
    real_y1 = int(round(y1 // ratio))
    real_x2 = int(round(x2 // ratio))
    real_y2 = int(round(y2 // ratio))

    return real_x1, real_y1, real_x2, real_y2

def display_image(result, img_path):
    #print(result)
    print(img_path)

    #img = cv2.imread(img_path)

    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    from PIL import Image
    import numpy as np

    im = np.array(Image.open(img_path), dtype=np.uint8)


    # Create figure and axes
    fig,ax = plt.subplots(1)

    # Display the image
    ax.imshow(im)

    #for coords in list(result)[0]:
    coords = list(result)[0]
    print(coords)
    x,y,w,h = coords[0], coords[1], coords[2], coords[3]
    print(x,y,w,h)
    # Create a Rectangle patch
    rect = patches.Rectangle((x,y),w,h,linewidth=1,edgecolor='r',facecolor='none')

    # Add the patch to the Axes
    ax.add_patch(rect)
    result_path = 'results_images/img.png'

    plt.savefig(result_path)
    print('result saved into ', result_path)



def predict_single_image(img_path, model_rpn, model_classifier_only, cfg, class_mapping):
    st = time.time()
    img = cv2.imread(img_path)
    if img is None:
        print('reading image failed.')
        exit(0)

    X, ratio = format_img(img, cfg)
    if K.image_dim_ordering() == 'tf':
        X = np.transpose(X, (0, 2, 3, 1))
    # get the feature maps and output from the RPN
    [Y1, Y2, F] = model_rpn.predict(X)

    # this is result contains all boxes, which is [x1, y1, x2, y2]
    result = roi_helpers.rpn_to_roi(Y1, Y2, cfg, K.image_dim_ordering(), overlap_thresh=0.7)
    # convert from (x1,y1,x2,y2) to (x,y,w,h)
    result[:, 2] -= result[:, 0]
    result[:, 3] -= result[:, 1]
    #display_image(list(result), img_path)
    bbox_threshold = 0.8
    #apply the spatial pyramid pooling to the proposed regions
    boxes = dict()
    for jk in range(result.shape[0] // cfg.num_rois + 1):
        rois = np.expand_dims(result[cfg.num_rois * jk:cfg.num_rois * (jk + 1), :], axis=0)
        if rois.shape[1] == 0:
            break
        if jk == result.shape[0] // cfg.num_rois:
            # pad R
            curr_shape = rois.shape
            target_shape = (curr_shape[0], cfg.num_rois, curr_shape[2])
            rois_padded = np.zeros(target_shape).astype(rois.dtype)
            rois_padded[:, :curr_shape[1], :] = rois
            rois_padded[0, curr_shape[1]:, :] = rois[0, 0, :]
            rois = rois_padded
        [p_cls, p_regr] = model_classifier_only.predict([F, rois])
#        print([p_cls, p_regr])
        for ii in range(p_cls.shape[1]):
            if np.max(p_cls[0, ii, :]) < 0.8 or np.argmax(p_cls[0, ii, :]) == (p_cls.shape[2] - 1):
                continue

            cls_num = np.argmax(p_cls[0, ii, :])
            if cls_num not in boxes.keys():
                boxes[cls_num] = []
            (x, y, w, h) = rois[0, ii, :]
            print(x,y,w,h)
            print(rois[0,ii,:])
            try:
                (tx, ty, tw, th) = p_regr[0, ii, 4 * cls_num:4 * (cls_num + 1)]
                tx /= cfg.classifier_regr_std[0]
                ty /= cfg.classifier_regr_std[1]
                tw /= cfg.classifier_regr_std[2]
                th /= cfg.classifier_regr_std[3]
                x, y, w, h = roi_helpers.apply_regr(x, y, w, h, tx, ty, tw, th)
            except Exception as e:
                print(e)
                pass
            boxes[cls_num].append(
                [cfg.rpn_stride * x, cfg.rpn_stride * y, cfg.rpn_stride * (x + w), cfg.rpn_stride * (y + h),
                 np.max(p_cls[0, ii, :])])

    # add some nms to reduce many boxes
    for cls_num, box in boxes.items():
        boxes_nms = roi_helpers.non_max_suppression_fast(box, overlap_thresh=0.5)
        boxes[cls_num] = boxes_nms
        print(class_mapping[cls_num] + ":")
        for b in boxes_nms:
            b[0], b[1], b[2], b[3] = get_real_coordinates(ratio, b[0], b[1], b[2], b[3])
            print('{} prob: {}'.format(b[0: 4], b[-1]))
    img = draw_boxes_and_label_on_image_cv2(img, class_mapping, boxes)
    print('Elapsed time = {}'.format(time.time() - st))
    #cv2.imshow('image', img)
    result_path = 'results_images/{}.png'.format(os.path.basename(img_path).split('.')[0])
    print('result saved into ', result_path)
    cv2.imwrite(result_path, img)
    cv2.waitKey(0)

    # bboxes = {}
    # probs = {}
    # for jk in range(result.shape[0]//cfg.num_rois + 1):
    #     ROIs = np.expand_dims(result[cfg.num_rois*jk:cfg.num_rois*(jk+1), :], axis=0)
    #     if ROIs.shape[1] == 0:
    #         break
    #
    #     if jk == result.shape[0]//cfg.num_rois:
	# 		#pad R
    #         curr_shape = ROIs.shape
    #         target_shape = (curr_shape[0],cfg.num_rois,curr_shape[2])
    #         ROIs_padded = np.zeros(target_shape).astype(ROIs.dtype)
    #         ROIs_padded[:, :curr_shape[1], :] = ROIs
    #         ROIs_padded[0, curr_shape[1]:, :] = ROIs[0, 0, :]
    #         ROIs = ROIs_padded
    #
    #     [P_cls, P_regr] = model_classifier_only.predict([F, ROIs])
    #
    #     for ii in range(P_cls.shape[1]):
    #         if np.max(P_cls[0, ii, :]) < bbox_threshold or np.argmax(P_cls[0, ii, :]) == (P_cls.shape[2] - 1):
    #             continue
    #
    #         cls_name = class_mapping[np.argmax(P_cls[0, ii, :])]
    #
    #         if cls_name not in bboxes:
    #             bboxes[cls_name] = []
    #             probs[cls_name] = []
    #
    #         (x, y, w, h) = ROIs[0, ii, :]
    #         cls_num = np.argmax(P_cls[0, ii, :])
    #         try:
    #
    #             (tx, ty, tw, th) = P_regr[0, ii, 4*cls_num:4*(cls_num+1)]
    #             tx /= C.classifier_regr_std[0]
    #             ty /= C.classifier_regr_std[1]
    #             tw /= C.classifier_regr_std[2]
    #             th /= C.classifier_regr_std[3]
    #             x, y, w, h = roi_helpers.apply_regr(x, y, w, h, tx, ty, tw, th)
    #         except:
    #             pass
    #         bboxes[cls_name].append([cfg.rpn_stride*x, cfg.rpn_stride*y, cfg.rpn_stride*(x+w), cfg.rpn_stride*(y+h)])
    #         probs[cls_name].append(np.max(P_cls[0, ii, :]))
    #
    #     all_dets = []
    #
    #     for key in bboxes:
    #         bbox = np.array(bboxes[key])
    #         new_boxes, new_probs = roi_helpers.non_max_suppression_fast(bbox, np.array(probs[key]), overlap_thresh=0.5)
    #         for jk in range(boxes_nms.shape[0]):
    #             (x1, y1, x2, y2) = new_boxes[jk,:]
    #             (real_x1, real_y1, real_x2, real_y2) = get_real_coordinates(ratio, x1, y1, x2, y2)
    #             cv2.rectangle(img,(real_x1, real_y1), (real_x2, real_y2), (int(class_to_color[key][0]), int(class_to_color[key][1]), int(class_to_color[key][2])),2)
    #             textLabel = '{}: {}'.format(key,int(100*new_probs[jk]))
    #             all_dets.append((key,100*new_probs[jk]))
    #             (retval,baseLine) = cv2.getTextSize(textLabel,cv2.FONT_HERSHEY_COMPLEX,1,1)
    #             textOrg = (real_x1, real_y1-0)
    #             cv2.rectangle(img, (textOrg[0] - 5, textOrg[1]+baseLine - 5), (textOrg[0]+retval[0] + 5, textOrg[1]-retval[1] - 5), (0, 0, 0), 2)
    #             cv2.rectangle(img, (textOrg[0] - 5,textOrg[1]+baseLine - 5), (textOrg[0]+retval[0] + 5, textOrg[1]-retval[1] - 5), (255, 255, 255), -1)
    #             cv2.putText(img, textLabel, textOrg, cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 0), 1)
    #
    # print('Elapsed time = {}'.format(time.time() - st))
    # print(all_dets)
    # cv2.imshow('img', img)
    # cv2.waitKey(0)
    # v2.imwrite('results_imgs/{}.png'.format(idx),img)

def predict(args_):
    path = args_.path
    with open('config.pickle', 'rb') as f_in:
        cfg = pickle.load(f_in)
    cfg.use_horizontal_flips = False
    cfg.use_vertical_flips = False
    cfg.rot_90 = False

    class_mapping = cfg.class_mapping
    if 'bg' not in class_mapping:
        class_mapping['bg'] = len(class_mapping)

    class_mapping = {v: k for k, v in class_mapping.items()}
    input_shape_img = (None, None, 3)
    input_shape_features = (None, None, 1024)

    img_input = Input(shape=input_shape_img)
    roi_input = Input(shape=(cfg.num_rois, 4))
    feature_map_input = Input(shape=input_shape_features)

    shared_layers = nn.nn_base(img_input, trainable=True)

    # define the RPN, built on the base layers
    num_anchors = len(cfg.anchor_box_scales) * len(cfg.anchor_box_ratios)
    rpn_layers = nn.rpn(shared_layers, num_anchors)
    classifier = nn.classifier(feature_map_input, roi_input, cfg.num_rois, nb_classes=len(class_mapping),
                               trainable=True)
    model_rpn = Model(img_input, rpn_layers)
    model_classifier_only = Model([feature_map_input, roi_input], classifier)

    model_classifier = Model([feature_map_input, roi_input], classifier)

    print('Loading weights from {}'.format(cfg.model_path))
    model_rpn.load_weights(cfg.model_path, by_name=True)
    model_classifier.load_weights(cfg.model_path, by_name=True)

    model_rpn.compile(optimizer='sgd', loss='mse')
    model_classifier.compile(optimizer='sgd', loss='mse')

    if os.path.isdir(path):
        for idx, img_name in enumerate(sorted(os.listdir(path))):
            if not img_name.lower().endswith(('.bmp', '.jpeg', '.jpg', '.png', '.tif', '.tiff')):
                continue
            print(img_name)

            predict_single_image(os.path.join(path, img_name), model_rpn,
                                 model_classifier_only, cfg, class_mapping)
    elif os.path.isfile(path):
        print('predict image from {}'.format(path))
        predict_single_image(path, model_rpn, model_classifier_only, cfg, class_mapping)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', '-p', default='images/000010.png', help='image path')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    predict(args)
