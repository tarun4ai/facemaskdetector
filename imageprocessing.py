import numpy as np
from utils.image import get_affine_transform
from utils.util import ctdet_post_process
import cv2
import utils.config as conf
import torch
from decode import ctdet_decode

mean = np.array(conf.config.mean, dtype=np.float32).reshape(1, 1, 3)
std = np.array(conf.config.std, dtype=np.float32).reshape(1, 1, 3)
num_classes = conf.config.num_classes
input_h = conf.config.input_h
input_w = conf.config.input_w
down_ratio = conf.config.down_ratio
topk = conf.config.topk
vis_threshold = conf.config.vis_threshold
dict = conf.config.dict

def pre_process( image, meta=None):
    height, width = image.shape[0:2]
    new_height = int(height)
    new_width = int(width)
    inp_height, inp_width = input_h, input_w
    c = np.array([new_width / 2., new_height / 2.], dtype=np.float32)
    s = max(height, width) * 1.0

    trans_input = get_affine_transform(c, s, 0, [inp_width, inp_height])
    resized_image = cv2.resize(image, (new_width, new_height))
    inp_image = cv2.warpAffine(resized_image, trans_input, (inp_width, inp_height), flags=cv2.INTER_LINEAR)
    inp_image = ((inp_image / 255. - mean) / std).astype(np.float32)

    images = inp_image.transpose(2, 0, 1).reshape(1, 3, inp_height, inp_width)
    meta = {'c': c, 's': s,
            'out_height': inp_height // down_ratio,
            'out_width': inp_width // down_ratio}
    return images, meta

def post_process(dets, meta, scale=1):
    dets = dets.detach().cpu().numpy()
    dets = dets.reshape(1, -1, dets.shape[2])
    dets = ctdet_post_process(
        dets.copy(), [meta['c']], [meta['s']],
        meta['out_height'], meta['out_width'], num_classes)
    for j in range(1, num_classes + 1):
        dets[0][j] = np.array(dets[0][j], dtype=np.float32).reshape(-1, 5)
        dets[0][j][:, :4] /= scale
    return dets[0]

def threshold_process(dets):
    result = []
    for j in range(1, num_classes + 1):
        for bbox in dets[j]:
            if bbox[4] > vis_threshold:
                result.append([bbox[:4], bbox[4], dict[str(j-1)] ])
    return result

def draw_bbox(image, bboxs):
    for bbox in bboxs:
        per_bbox, conf, label = bbox
        per_bbox = per_bbox.astype(np.int)
        color = [0,0,255] if label=='face' else [0,255,0]
        cv2.rectangle(image, (per_bbox[0], per_bbox[1]), (per_bbox[2], per_bbox[3]), color, 2)
        txt = '{}{:.1f}'.format(label, conf)
        font = cv2.FONT_HERSHEY_SIMPLEX
        cat_size = cv2.getTextSize(txt, font, 0.5, 2)[0]
        cv2.rectangle(image,
                    (per_bbox[0], per_bbox[1] - cat_size[1] - 2),
                    (per_bbox[0] + cat_size[0], per_bbox[1] - 2), color, -1)
        cv2.putText(image, txt, (per_bbox[0], per_bbox[1] - 2),
                    font, 0.5, (0, 0, 0), thickness=1, lineType=cv2.LINE_AA)
    return image

def process(img,sess):
    input_name = sess.get_inputs()[0].name
    label_name1 = sess.get_outputs()[0].name
    label_name2 = sess.get_outputs()[1].name
    label_name3 = sess.get_outputs()[2].name

    image = img.copy()
    img,meta = pre_process(img)
    hm,reg,wh = sess.run([label_name1,label_name2,label_name3],{input_name:img})

    hm = torch.from_numpy(hm).sigmoid_()
    wh = torch.from_numpy(wh)
    reg = torch.from_numpy(reg)

    dets = ctdet_decode(hm, wh, reg=reg, K=topk)
    dets = post_process(dets, meta)
    dets = threshold_process(dets)

    im = draw_bbox(image,dets)
    return im