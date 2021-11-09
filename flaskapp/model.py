import torch
import os, sys
import cv2
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import skimage.measure
import time
import math
from tqdm import tqdm

# # Test Model

from yolov5.models.experimental import attempt_load
from yolov5.utils.general import check_img_size, non_max_suppression, apply_classifier, scale_coords
from yolov5.utils.plots import colors
from yolov5.utils.torch_utils import select_device, load_classifier

import base64
import json

# Image stretching (constrast)
def clip_stretch(image, cmin, cmax):
    image = np.clip(image, cmin, cmax)
    image = (image - cmin) / (cmax - cmin) * 255.0
    return image.astype('uint8')

def mask_img(img):
    
    # Convert to grayscale
    mask = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    
    # Increase constrast + small blur to interpolate values
    mask = clip_stretch(mask, 175,220)
    mask = cv2.blur(mask,(3,3))

    # Blur/Clear middle portion
    h, w = mask.shape[:2]
    h, w= h//2, w//2
    midsize = 400
    mask[h-midsize:h+midsize,w-midsize:w+midsize] = cv2.blur(mask[h-midsize:h+midsize,w-midsize:w+midsize], (midsize, midsize))

    # Find edges
    mask = cv2.Canny(mask, 50, 50)

    # Reduce resolution by 6 times with max pixel value in attempts to join up boundaries
    origin_h, origin_w = mask.shape
    mask = skimage.measure.block_reduce(mask, (6,6), np.max)

    # Flood fill from center
    h, w = mask.shape
    mask_pad =  np.pad(mask, 1, 'minimum')
    cv2.floodFill(mask, mask_pad, seedPoint=(w//2,h//2-20), newVal=150)
    mask = np.where(mask == 150, 255, 0).astype('uint8')
    
    # Perform image closing to fill gaps
    mask = cv2.dilate(mask, np.ones((15,15)).astype('uint8'),iterations=1)
    mask = cv2.erode(mask, np.ones((14,14)).astype('uint8'),iterations=1)
   
    # Resize image back
    mask = cv2.resize(mask, (origin_w, origin_h))
    mask = np.where(mask==0,0,255).astype('uint8')

    # Perform image closing once more
    mask = cv2.dilate(mask, np.ones((40,40)).astype('uint8'),iterations=1)
    mask = cv2.erode(mask, np.ones((30,30)).astype('uint8'),iterations=1)
    mask = cv2.blur(mask,(15,15))

    # Covert to 0-1
    mask = mask / 255.0

    img = img.astype('float64')
    for channel in range(3):
        img[:,:,channel] *= mask
    img = img.astype('uint8')

    return img

def encodebase64(img):
    return base64.b64encode(cv2.imencode('.jpg', img)[1]).decode('utf8')


def letterbox(img, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=False, stride=32):
    # Resize and pad image while meeting stride-multiple constraints
    shape = img.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better test mAP)
        r = min(r, 1.0)

    # Compute padding
    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding
    elif scaleFill:  # stretch
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    return img, ratio, (dw, dh)

class LoadImages_2:  # for inference
    def __init__(self, pic_dict_values, img_size=640, stride=32):
        ni = len(pic_dict_values)
        self.pic_dict_values = pic_dict_values
        self.img_size = img_size
        self.stride = stride
        self.nf = ni   # number of files
        self.video_flag = [False] * ni 
        self.mode = 'image'
        self.cap = None

    def __iter__(self):
        self.count = 0
        return self

    def __next__(self):
        if self.count == self.nf:
            raise StopIteration

        if self.video_flag[self.count]:
            # Read video
            self.mode = 'video'

        else:
            # Read image
            img0 = self.pic_dict_values[self.count]
            img0 = cv2.cvtColor(img0, cv2.COLOR_BGR2RGB)
            self.count += 1
            assert img0 is not None, 'Image Not Found '
            print(f'image {self.count}/{self.nf}: ', end='')

        # Padded resize
        img = letterbox(img0, self.img_size, stride=self.stride)[0]

        # Convert
        img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
        img = np.ascontiguousarray(img)

        return img, img0

    def __len__(self):
        return self.nf  # number of files

def plot_one_box(x, im, color=(128, 128, 128), label=None, line_thickness=3):
    # Plots one bounding box on image 'im' using OpenCV
    assert im.data.contiguous, 'Image not contiguous. Apply np.ascontiguousarray(im) to plot_on_box() input image.'
    tl = line_thickness or round(0.002 * (im.shape[0] + im.shape[1]) / 2) + 1  # line/font thickness
    c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
    cv2.rectangle(im, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)
    if label:
        tf = max(tl - 1, 1)  # font thickness
        t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
        c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
        cv2.rectangle(im, c1, c2, color, -1, cv2.LINE_AA)  # filled
        cv2.putText(im, label, (c1[0], c1[1] - 2), 0, tl / 3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)


def run_model(datafiles, mask = False, fast = False, IMAGE_SIZE = 4032):

    # Device to use (e.g. "0", "1", "2"... or "cpu")
    DEVICE = "cpu"

    # WEIGHTS = "weights/best_2.pt" # best weights: baseline for augmented train
    # # # to be switched to the tuned one
    if fast:
        WEIGHTS = "weights/best_s.pt"
    else:
        WEIGHTS = "weights/best_l.pt"

    # masked_bool_dict = {} # if we can get a list of which images to mask and not mask
    # and store it as dict: key = img_name , value = boolean where true = mask
    masked_img_dict = {}
    raw_img_dict = {}
    shape_dict = {}

    for datafile in datafiles:
        im = datafile['img']
        item_name = datafile['filename']
        raw_img_dict[item_name] = im
        shape_dict[item_name] = im.shape
        # masked_bool_dict[item_name] = True # default = mask

    for img_name in raw_img_dict:
        raw_img = raw_img_dict[img_name].copy()
        # to_mask = masked_bool_dict[img_name]
        # if to_mask: # or if wanna just check for all
        if mask:  # then change to this line instead
          out_img = mask_img(raw_img) # mask it
        else:
          out_img = raw_img # no masking
        masked_img_dict[img_name] = out_img

    # init model params
    device = select_device(DEVICE) 
    model = attempt_load(WEIGHTS, map_location=device) # weights are used here
    stride = 32
    imgsz = check_img_size(IMAGE_SIZE, s=stride)
    names = ['Fertilised Egg', 'Unfertilised Egg', 'Fish Larvae', 'Unidentifiable']
    dataset = LoadImages_2(np.asarray(list(masked_img_dict.values())), img_size=IMAGE_SIZE, stride=stride)

    # Model Inference
    img_num = 0
    empty_df = []
    no_pred_imgs = []
    start = time.time()
    count = 0
    total_det = 0
    for img_og, im0s in dataset:
        img = torch.from_numpy(img_og).to(device).float()
        img /= 255.0  # normalize image
        path = list(raw_img_dict.keys())[count]
        if img.ndimension() == 3:
            img = img.unsqueeze(0) # Include batch dimension
            pred = model(img)[0]
            pred_copy1 = pred.clone()
            # To group multiple bounding boxes into 1 based on IOU
            pred = non_max_suppression(pred, conf_thres= 0.45, iou_thres=0.15, max_det=1000)
            pred_copy = pred.copy()
            # Process detections
            for i, det in enumerate(pred):  # detections per image
                s, im0, frame = '', im0s.copy(), getattr(dataset, 'frame', 0)
                det_copy = det.clone()
                print("Number of Detections =", len(det))
                total_det += len(det)
                if len(det): # skip images w empty labels
                    # Rescale boxes from img_size to im0 size
                    det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()
                    px = pd.DataFrame(det.numpy())
                    px['pic_no'] = path
                    px['x1'] = px[0]
                    px['x2'] = px[2]
                    px['y1'] = px[1]
                    px['y2'] = px[3]
                    px['x'] = ((px[0] + px[2])/2)/img_og.shape[2] # x: midpoint
                    px['y'] = ((px[1] + px[3])/2)/img_og.shape[1] # y: midpoint
                    px['w'] = abs((px[0] - px[2])/2)/img_og.shape[2] # width
                    px['h'] = abs((px[1] - px[3])/2)/img_og.shape[1] # height
                    px['predicted_class'] = px[5]
                    px['confidence'] = px[4]
                    empty_df.append(px)
                else:
                    no_pred_imgs.append(path)
            
            count +=1

    # Redraw bounding boxes on original image
    # final_dict = {}
    empty_count_df = []
    empty_file_df = []
    annotated_img_list = []
    if len(empty_df):
        px2_df = pd.concat(empty_df, ignore_index=True)
        for pic_no in raw_img_dict:
            img_og = raw_img_dict[pic_no].copy() # original image
            img = torch.from_numpy(img_og).to(device).float()
            img /= 255.0  # normalize image
            if img.ndimension() == 3:
                img = img.unsqueeze(0) # Include batch dimension
                px_coords = px2_df[px2_df['pic_no'] == pic_no][['x1', 'y1', 'x2', 'y2', 'confidence', 'predicted_class']]
                px_tensor_coords = torch.tensor(px_coords.values)
                pred = [px_tensor_coords]                                              # remove this and above two lines for non-nms
                px_coords['counts'] = 1
                px_coords['pic_no'] = pic_no
                cols = {0: 'Fertilised Egg', 1: 'Unfertilised Egg', 2: 'Fish Larvae', 3: 'Unidentifiable'}
                count_df = px_coords.pivot_table('counts', ['pic_no'], 'predicted_class', aggfunc = [np.sum], fill_value=0).reset_index()
                count_df = count_df.rename(columns = cols)
                empty_count_df.append(count_df)
                for i, det in enumerate(pred):  # detections per image
                    if len(det):
                        # Rescale boxes from img_size to im0 size
                        for c in det[:, -1].unique():
                            n = (det[:, -1] == c).sum()  # detections per class
                            s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string  
                        for *xyxy, conf, cls in reversed(det):
                            c = int(cls)  # integer class
                            label = f'{names[c]} {conf:.2f}'
                            plot_one_box(xyxy, img_og, label= None, #label
                                        color=colors(c, True), line_thickness=3)
                # Save image
                annotated_img_list.append(img_og)

                # output list of dict format
                px_dev = px2_df[px2_df['pic_no'] == pic_no][['x', 'y', 'w', 'h','predicted_class', 'confidence']] # retrieve bb coords
                px_dev['x'] = px_dev.apply(lambda x: round(x['x'], 6), axis = 1)
                px_dev['y'] = px_dev.apply(lambda x: round(x['y'], 6), axis = 1)
                px_dev['w'] = px_dev.apply(lambda x: round(x['w'], 6), axis = 1)
                px_dev['h'] = px_dev.apply(lambda x: round(x['h'], 6), axis = 1)
                px_dev['confidence'] = px_dev.apply(lambda x: round(x['confidence'], 6), axis = 1)
                px_dev['bounding_box'] = px_dev[['x','y','w','h']].values.tolist()
                px_dev['predicted_class'] = px_dev.apply(lambda x: int(x['predicted_class']), axis = 1)
                px_dev = px_dev.iloc[:, 4:7]
                bb_vals = px_dev.to_json(orient="records")
                file_data = {}
                file_data['filename'] = pic_no # image/file name
                file_data['image_base64'] = encodebase64(raw_img_dict[pic_no]) # orginal image np array > can convert at this step to base64 image
                # file_data['image'] = raw_img_dict[pic_no] # orginal image np array > can convert at this step to base64 image
                # annotated_img_list.append(final_dict[pic_no])
                # no word labels, maybe need a legend or sth
                # 0: Fertilized Egg - Red
                # 1: Unfertilized Egg - Pink
                # 2: Fish Larva - Orange
                # 3: Unidentifiable Object - Yellow
                file_data['predictions'] = json.loads(bb_vals) # bounding boxes
                empty_file_df.append(file_data)

    else: # no predictions at all for all images

        for pic_no in raw_img_dict:
            file_data = {}
            file_data['filename'] = pic_no # image/file name
            file_data['image_base64'] = encodebase64(raw_img_dict[pic_no]) # orginal image np array > can convert at this step to base64 image
            # file_data['image'] = raw_img_dict[pic_no] # orginal image np array > can convert at this step to base64 image
            annotated_img_list.append(raw_img_dict[pic_no])
            file_data['predictions'] = []
            empty_file_df.append(file_data)


    counts = {}

    for d in empty_file_df:
        fn = d['filename']
        if fn not in counts:
            counts[fn] = [0,0,0,0]
        for pred in d['predictions']:
            counts[fn][pred['predicted_class']]+=1
    for fn in no_pred_imgs:
        counts[fn] = [0,0,0,0]
    final_counts = pd.DataFrame.from_dict(counts,orient='index').reset_index()
    final_counts.columns = ['pic_name', 'Fertilised Egg', 'Unfertilised Egg', 'Fish Larvae', 'Unidentifiable']

    print(final_counts)

    #     # Count outputs (optional i guess):
    #     px2_df['counts'] = 1
    #     # cols = {0: 'Fertilised Egg', 1: 'Unfertilised Egg', 2: 'Fish Larvae', 3: 'Unidentifiable'}
    #     cols = ['pic_no', 'Fertilised Egg', 'Unfertilised Egg', 'Fish Larvae', 'Unidentifiable']
    #     px_df1 = px2_df.pivot_table('counts', ['pic_no'], 'predicted_class', aggfunc = [np.sum], fill_value=0, dropna=False).reset_index().droplevel(0, axis=1)
    #     print(px_df1)
    #     # final_counts = px_df1.rename(columns = cols)
    #     px_df1.columns = cols
    #     if no_pred_imgs:
    #         no_pred_imgs = pd.DataFrame(map(lambda x: [x,0,0,0,0], no_pred_imgs))
    #         no_pred_imgs.columns = cols
    #         # print(final_counts)
    #         print(no_pred_imgs)
    #     final_counts = pd.concat([px_df1, no_pred_imgs], ignore_index=True)
    #     print(final_counts)

    # else: # no predictions at all for all images

    #     final_counts = pd.DataFrame(map(lambda x: [x,0,0,0,0], no_pred_imgs))
    #     final_counts.columns = cols

    #     # empty_data_dict = {}
    #     # for pic_no in raw_img_dict:
    #     #     img_og = raw_img_dict[pic_no].copy() # original image
    #     #     file_data = {}
    #     #     file_data['filename'] = pic_no # image/file name
    #     #     file_data['image'] = encodebase64(raw_img_dict[pic_no]) # orginal image np array > can convert at this step to base64 image
    #     #     # file_data['image'] = raw_img_dict[pic_no] # orginal image np array > can convert at this step to base64 image
    #     #     annotated_img_list.append(raw_img_dict[pic_no])
    #     #     file_data['predictions'] = []
    #     #     empty_file_df.append(file_data)
    #     #     empty_data_dict[pic_no] = [0, 0, 0, 0]

    #     # final_counts = pd.DataFrame.from_dict(empty_data_dict, orient='index', columns=['Fertilised Egg', 'Unfertilised Egg', 'Fish Larvae', 'Unidentifiable'])
    #     print(final_counts)
    
    return empty_file_df, annotated_img_list, final_counts

