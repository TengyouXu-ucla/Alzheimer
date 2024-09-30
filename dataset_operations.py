from torch.utils.data import Dataset
import albumentations as A
import random
from torchvision import transforms as T
import os
import cv2
import numpy as np
import re
from utils import im_to_txt_path, imread, read_label, isfile
import torch
import shutil
import matplotlib.pyplot as plt
import torchvision
import logging
import traceback
from timer import *

def delete_all_files_and_dirs_inside(dir_path):
    # Check if the directory exists
    if not os.path.exists(dir_path):
        raise ValueError(f"The directory {dir_path} does not exist.")
    
    # Iterate over all the contents of the directory
    for filename in os.listdir(dir_path):
        file_path = os.path.join(dir_path, filename)
        
        # Check if it's a file or directory and delete it
        if os.path.isfile(file_path) or os.path.islink(file_path):
            os.remove(file_path)
            print(f"Deleted file: {file_path}")
        elif os.path.isdir(file_path):
            shutil.rmtree(file_path)
            print(f"Deleted directory: {file_path}")
    
    
def reset_dir(tile_dir):
    '''Reset the tile dir to include only two given sub directories, and make sure they are empty'''
    
    if not os.path.exists(tile_dir):
        os.makedirs(tile_dir)
    if os.path.exists(tile_dir):
        delete_all_files_and_dirs_inside(tile_dir)
    return
    
def get_img_paths(img_dir, test_ratio = 0.15, val_ratio = 0.15):
    img_paths = []
    for root, dirs, files in os.walk(img_dir):
            for file in files:
                if file.lower().endswith(('.jpg','.png')):
                    img_path = os.path.join(root, file)
                    img_paths.append(img_path)            


    test_roi_paths = random.sample(img_paths, int(len(img_paths) * test_ratio))

    print(f"Number of test images: {len(test_roi_paths)}")

    for roi_path in test_roi_paths:
        img_paths.remove(roi_path)

    with open(img_dir + "test_rois.txt", "w") as f:
        for roi_path in test_roi_paths:
            f.write(roi_path + "\n")

    val_roi_paths = random.sample(img_paths, int(len(img_paths) * val_ratio / (1 - test_ratio)))

    print(f"Number of val images: {len(val_roi_paths)}")

    for roi_path in val_roi_paths:
        img_paths.remove(roi_path)
        
    train_roi_paths = img_paths

    with open(img_dir + "train_rois.txt", "w") as f:
        for roi_path in train_roi_paths:
            f.write(roi_path + "\n")
    with open(img_dir + "val_rois.txt", "w") as f:
        for roi_path in val_roi_paths:
            f.write(roi_path + "\n")
            
    print(f"Number of train images: {len(train_roi_paths)}")

    return train_roi_paths, val_roi_paths, test_roi_paths

def show_img_with_labels(img_fp, label_fp):
    img = imread(img_fp)
    img_size = img.shape[1]
    labels = []
    if isfile(label_fp):
        labels = read_label(label_fp)
        for label in labels:
            # print(label)
            xc, yc, w, h = label[1], label[2], label[3], label[4]
            x1 = int((xc - w/2) * img_size)
            y1 = int((yc - h/2) * img_size)
            x2 = int((xc + w/2) * img_size)
            y2 = int((yc + h/2) * img_size)
            # print(x1, y1, x2, y2)
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
    print('num of labels: ', len(labels))
    plt.imshow(img)
    plt.show()
    return

def randomly_show_img_with_labels(train_tiles_df):
    crop_paths = []
    for tile in train_tiles_df['fp']:
        if isfile(tile):
            crop_paths.append(tile)
                    
    random_items = random.sample(crop_paths, 5)

    for img in random_items:
        label_fp = im_to_txt_path(img)
        show_img_with_labels(img, label_fp)
    return 


def calculate_mean_std(roi_paths,target_dir):
    input_imgs = roi_paths
    # Path to your dataset
    batch_size = 1000  # Adjust batch size based on your memory capacity
    num_batches = len(input_imgs) // batch_size + 1

    # Initialize variables to calculate mean and standard deviation
    sum_pixel_values = np.zeros(3, np.float64)
    sum_squared_pixel_values = np.zeros(3, np.float64)
    total_pixels = 0

    # Batch processing
    for i in range(num_batches + 1):
        batch_files = input_imgs[i * batch_size: min((i + 1) * batch_size, len(input_imgs))]
        
        for image_file in batch_files:
            image = cv2.imread(image_file)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = image.astype(np.float64) / 255.0  # Scale pixel values to [0, 1]
            
            sum_pixel_values += np.sum(image, axis=(0, 1))
            sum_squared_pixel_values += np.sum(image ** 2, axis=(0, 1))
            total_pixels += image.shape[0] * image.shape[1]

    # Calculate the mean
    mean = sum_pixel_values / total_pixels

    # Calculate the variance and standard deviation
    variance = (sum_squared_pixel_values / total_pixels) - (mean ** 2)
    std = np.sqrt(variance)

    print("Mean:", mean)
    print("Standard Deviation:", std)

    with open(target_dir + 'mean_std.txt', 'w') as f:
        f.write(f"Mean: {mean}\n")
        f.write(f"Standard Deviation: {std}\n")




def center_to_unorm_corner_coord(center_arr, img_size = 512):
  xc, yc, w, h = center_arr
  xc, yc, w, h = xc * img_size, yc * img_size, w * img_size, h * img_size
  x1, y1, x2, y2 = xc - w/2, yc - h/2, xc + w/2, yc + h/2
  return [x1, y1, x2, y2]

def load_inputs(tile_dir):
    '''Load the inputs from the tile directory'''
    # input_imgs = []
    # input_targets = []
    inputs = []

    for root, dirs, files in os.walk(tile_dir + 'images/'):
        for file in files:
            if file.lower().endswith(('.png')):
                
                
                boxes = []
                labels = []
                
                if os.path.exists(im_to_txt_path(root + file)): 
                    inputf = open(im_to_txt_path(root + file))
                    for line in inputf.readlines():
                        temp = line.split(' ')
                        temp = [float(i) for i in temp]
                        labels.append(int(temp[0]))
                        x1, y1, x2, y2 = center_to_unorm_corner_coord(temp[1:])
                        boxes.append([x1, y1, x2, y2])
                    labels = torch.tensor(labels, dtype = torch.int64)
                    boxes = torch.tensor(boxes, dtype = torch.float64)
                else:
                    labels = torch.empty(0, dtype = torch.int64)
                    boxes = torch.empty((0,4), dtype = torch.float64)
                
                
                # input_imgs.append(root + file)
                # print(root+file)
                # input_targets.append({'boxes': boxes, 'labels': labels})
                inputs.append({'image': root + file, 'target': {'boxes': boxes, 'labels': labels}})
                        
    # print(len(input_imgs))
    # print(len(input_targets))
    print(len(inputs))
    return inputs




def read_mean_std(tile_dir):
    with open(tile_dir + 'mean_std.txt') as f:
        lines = f.readlines()
        
    mean = []
    std = []

    for line in lines:
        if line.startswith("Mean: "):
            # Extract the numbers using regex
            numbers_str = re.findall(r'\[([0-9.\s]+)\]', line)[0]
            # Convert the string of numbers to a list of floats
            mean = np.fromstring(numbers_str, sep=' ').tolist()
        if line.startswith("Standard Deviation: "):
            # Extract the numbers using regex
            numbers_str = re.findall(r'\[([0-9.\s]+)\]', line)[0]
            # Convert the string of numbers to a list of floats
            std = np.fromstring(numbers_str, sep=' ').tolist()
    print('mean: ', mean)
    print('std: ', std)
    return mean, std

def generate_tiles(roi_fps, roi_label_dir=None, tile_label_dir = None, tile_dir=None, tile_size = 512, fill=(114, 114, 114), box_thr = 0.8, notebook: bool = False,
    negative_ratio: float = 1.0, negative_label_ratio: float = 1.0, negatives_only = False, negatives_boxes = False, to_csv = True):
    if os.path.exists(tile_dir): 
        tiles_df = random_crop_roi_wrapper(
            roi_fps=roi_fps,
            roi_label_dir=roi_label_dir,
            tile_label_dir=tile_label_dir,
            save_dir=tile_dir,
            crop_size=tile_size,
            fill=fill,
            box_thr=box_thr,
            notebook=notebook,
            negative_ratio=negative_ratio,
            negative_label_ratio=negative_label_ratio,
            negatives_only = negatives_only,
            negatives_boxes = negatives_boxes
        )
        if to_csv:
            tiles_df.to_csv(tile_dir + 'tiles dataframe.csv') # Output the dataframe to a .csv file for data analysis
        return tiles_df

def get_transform(mean,std,type : str):
    random.seed(42)
    if type == 'A':
        transform = A.Compose([
            A.RandomRotate90(),
                A.Flip(),
                A.Transpose(),
                A.GaussNoise(),
                A.OneOf([
                    A.MotionBlur(p=.2),
                    A.MedianBlur(blur_limit=3, p=0.1),
                    A.Blur(blur_limit=3, p=0.1),
                ], p=0.2),
                A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.2, rotate_limit=45, p=0.2),
                A.OneOf([
                    A.OpticalDistortion(p=0.3),
                    A.GridDistortion(p=.1),
                ], p=0.2),
                A.OneOf([
                    A.CLAHE(clip_limit=2),
                    A.RandomBrightnessContrast(),
                ], p=0.3),
                A.HueSaturationValue(p=0.3),
                A.RandomCrop(256,256,False,0.3),
                A.Resize(512,512),
            A.Normalize(mean=mean, std=std, max_pixel_value=255.0, p=1.0),
            # A.ToTensorV2(p=1.0)
            
        ], bbox_params=A.BboxParams(format='pascal_voc', min_area=1024, min_visibility=0.1, label_fields=['class_labels']))
        
        return transform
    elif type == 'T':
        transform = T.Compose([
            T.ToTensor(),
            T.Normalize(mean=mean, std=std ),
            
        ])
        return transform
    else: 
        print("Invalid type")
        return None

def custom_collate_fn(batch):
    return list(batch)

class InputDataset(Dataset):

    def __init__(self, inputs_list, mean, std, trans:str = 'T'):
       
        self.inputs_list = inputs_list
        self.trans = trans
        self.mean = mean
        self.std = std

    def __len__(self):
        return len(self.inputs_list)

    def __getitem__(self, idx):
        # print(self.inputs_list[idx]['image'])
        self.transform = get_transform(self.mean, self.std, self.trans)
        
        
        img = imread(self.inputs_list[idx]['image'])
        
        boxes = self.inputs_list[idx]['target']['boxes']
        labels = self.inputs_list[idx]['target']['labels']
        
        if self.trans == 'T':
            img = self.transform(img)
        elif self.trans == 'A':
            augmented = self.transform(image = img, bboxes = boxes, class_labels = labels)
            img = torch.tensor(augmented['image'],dtype=torch.float32).permute(2,0,1)
            boxes = torch.tensor(augmented['bboxes'], dtype = torch.float32)
            labels = torch.tensor(augmented['class_labels'], dtype = torch.int64)
        else:
            img = T.ToTensor()(img)

        if boxes.shape[0] == 0:
            boxes = torch.empty((0,4), dtype = torch.float64)
            labels = torch.empty(0, dtype = torch.int64)
        targets = {'boxes': boxes, 'labels': labels}
        
        
        return {'image': img, 'target': targets}
    
def calculate_iou(box1, box2):
    """
    Calculate the Intersection over Union (IoU) of two bounding boxes.
    
    Parameters:
    box1, box2: Arrays or lists in the format [x_min, y_min, x_max, y_max]
    
    Returns:
    float: IoU value
    """
    x1_min, y1_min, x1_max, y1_max = box1
    x2_min, y2_min, x2_max, y2_max = box2
    
    # Calculate the intersection coordinates
    x_min = max(x1_min, x2_min)
    y_min = max(y1_min, y2_min)
    x_max = min(x1_max, x2_max)
    y_max = min(y1_max, y2_max)
    
    # Calculate the intersection area
    intersection_area = max(0, x_max - x_min) * max(0, y_max - y_min)
    
    # Calculate the area of both boxes
    box1_area = (x1_max - x1_min) * (y1_max - y1_min)
    box2_area = (x2_max - x2_min) * (y2_max - y2_min)
    
    # Calculate the union area
    union_area = box1_area + box2_area - intersection_area
    
    # Calculate the IoU
    iou = intersection_area / union_area
    
    return iou

# Configure logging
# logging.basicConfig(level=logging.DEBUG)


def detect_on_roi(model, roi_fp, window_size = 512, stride = 256, 
                  fill = [114, 114, 114], intersect_thr = 0.5, score_thr = 0.7, nms_thr = 0.5, false_neg_label_dir = None, false_pos_label_dir = None, plot = False, false_positive = True, false_negative = False):
    model.eval()
    timer = Timer()
    
    false_pos_label_path = false_pos_label_dir + roi_fp.split('/')[-1].split('.')[0] + '.txt'
    false_neg_label_path = false_neg_label_dir + roi_fp.split('/')[-1].split('.')[0] + '.txt'
    # if isfile(false_pos_label_path):
    if false_positive:
        false_pos_label_file = open(false_pos_label_path, 'w')
    if false_negative:
        false_neg_label_file = open(false_neg_label_path, 'w')
    timer.start()

    if plot == True:
        pred_boxes_on_roi = []

        
    logging.debug(f"Processing {roi_fp}")
    # load groundtruth labels
    true_boxes = []
    
    true_labels = []
    if isfile(im_to_txt_path(roi_fp)):
        with open(im_to_txt_path(roi_fp), 'r') as f:
            for line in f:
                temp_label = [int(x) for x in line.strip().split(' ')]
                true_boxes.append([temp_label[1], temp_label[2], temp_label[3], temp_label[4]])
                true_labels.append(temp_label[0])
    true_boxes_detected = [0] * len(true_boxes)
    roi = cv2.imread(roi_fp)
    
    timer.stop()
    # print(f"     Time to load roi:", timer.elapsed_time())
    
    if roi is None:
        logging.error(f"Failed to read {roi_fp}")
        return (0, 0)
    
    roi = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)/255
    roi_w = roi.shape[1]
    roi_h = roi.shape[0]
    roi = cv2.copyMakeBorder(roi, 0, window_size - roi_h % window_size, 0, window_size - roi_w % window_size, cv2.BORDER_CONSTANT, value=fill)
    
    true_boxes_map = torch.zeros((roi_h, roi_w))
    true_index_box = torch.zeros((roi_h, roi_w))
    
    for i, true_box in enumerate(true_boxes):
        x1, y1, x2, y2 = true_box
        x1,y1,x2,y2 = int(x1), int(y1), int(x2), int(y2)
        true_boxes_map[y1:y2, x1:x2] = (y2-y1)*(x2-x1)
        true_index_box[y1:y2, x1:x2] = i + 1
    
    # print('     roi width: ', roi_w, ' roi height: ', roi_h)
    start_x = 0
    
    while start_x <= roi_w:
        windows = []
        offsets = []
        
        # print('     ', start_x)
        start_y = 0
        while start_y <= roi_h: # slide the window from top left to bottom right
            end_x = int(start_x + window_size)
            end_y = int(start_y + window_size)
            window = roi[start_y:end_y, start_x:end_x]
            window = torch.tensor(window).permute(2, 0, 1).float().to('cuda')
            
            resize_transform = T.Resize((512, 512))
            window = resize_transform(window.unsqueeze(0)).squeeze(0)
            
            windows.append(window)
            offsets.append([start_x, start_y])
            
            start_y += stride
        # print('     number of windows: ', len(windows))
            
        preds = model(windows)
        # print(preds)
        # timer.stop()
        # print("     Time to predict ", len(windows), " windows: ", timer.elapsed_time())
        
        # window_timer = Timer()
        sub_timer = Timer()
        sub_timer.start()
        for idx, pred in enumerate(preds):
            # window_timer.start()
            
            nms_timer = Timer()
            offset = offsets[idx]
            nms_timer.start()
            filtered_idx = torchvision.ops.nms(pred['boxes'], pred['scores'], nms_thr)
            nms_timer.stop()
            # print("     Time to perform NMS on current window: ", nms_timer.elapsed_time())
            
            pred_boxes = pred['boxes'][filtered_idx]
            pred_scores = pred['scores'][filtered_idx]
            
            # print('     number of true boxes: ', len(pred_boxes))
            # print('     number of all boxes: ', len(pred['boxes']))
            
            for score, box in zip(pred_scores,pred_boxes):
                if score < score_thr:
                    continue
                
                correct = False
                
                # print('box: ', box)
                pred_box_x1 = box[0]
                pred_box_y1 = box[1]
                pred_box_x2 = box[2]
                pred_box_y2 = box[3]
                # print('     orig_pred_box: ', box )
                pred_box = [pred_box_x1 + offset[0], pred_box_y1 + offset[1], 
                            pred_box_x2 + offset[0], pred_box_y2 + offset[1]]
                # print('     pred_box: ', pred_box)
                if plot:
                    pred_boxes_on_roi.append(pred_box)
                intersect_area = (true_boxes_map[int(pred_box[1]):int(pred_box[3]), int(pred_box[0]):int(pred_box[2])]>0).sum().item()
                pred_area = (pred_box[3] - pred_box[1]) * (pred_box[2] - pred_box[0])
                if intersect_area > 0:
                    obj_area = torch.max(true_boxes_map[int(pred_box[1]):int(pred_box[3]), int(pred_box[0]):int(pred_box[2])]).item()
                    if intersect_area >= (obj_area + pred_area - intersect_area) * intersect_thr:
                        correct = True
                        index = int(torch.max(true_index_box[int(pred_box[1]):int(pred_box[3]), int(pred_box[0]):int(pred_box[2])]).item()) - 1   
                        true_boxes_detected[index] = 1                 
                if false_positive and not correct:
                    false_pos_label_file.write(f"{roi_fp} {pred_box[0]} {pred_box[1]} {pred_box[2]} {pred_box[3]} {score}\n")
        
        sub_timer.stop()
        # print("     Time to process current row of ", len(windows) ," windows: ", sub_timer.elapsed_time())
        start_x += stride
    timer.stop()
    if false_negative: 
        for i, _ in enumerate(true_boxes_detected):
            true_box = true_boxes[i]
            true_label = true_labels[i]
            if true_boxes_detected[i] == 0:
                false_neg_label_file.write(f"{true_label} {true_box[0]} {true_box[1]} {true_box[2]} {true_box[3]}\n")
            # window_timer.stop()
            # print("         Time to process current window: ", window_timer.elapsed_time())
    print("     Time to process current roi: ", timer.elapsed_time())

    # if plot:
    #     plot_img_with_boxes(roi_fp, pred_boxes_on_roi, true_boxes)
    
    false_pos_label_file.close()
    false_neg_label_file.close()
    num_false_pos_preds = 0
    with open(false_pos_label_path) as f:
        num_false_pos_preds = len(f.readlines())
    f.close()
    if plot:
        return pred_boxes_on_roi
    
    return sum(true_boxes_detected), len(true_boxes_detected), num_false_pos_preds

def count_preds_on_roi(model, roi_fp, window_size = 512, stride = 256, fill = [114, 114, 114],nms_thr=0.1, score_thr=0.5):
    model.eval()
    total_preds  = 0
    
    if roi_fp is None:
        logging.error(f"Failed to read {roi_fp}")
        return (0, 0)
    
    roi = imread(roi_fp)
    roi_w = roi.shape[1]
    roi_h = roi.shape[0]
    roi = cv2.copyMakeBorder(roi, 0, window_size - roi_h % window_size, 0, window_size - roi_w % window_size, cv2.BORDER_CONSTANT, value=fill)
    
    print('     roi width: ', roi_w, ' roi height: ', roi_h)
    start_x = 0
    
    while start_x <= roi_w:
        windows = []
        
        # print('     ', start_x)
        start_y = 0
        while start_y <= roi_h: # slide the window from top left to bottom right
            end_x = int(start_x + window_size)
            end_y = int(start_y + window_size)
            window = roi[start_y:end_y, start_x:end_x]
            window = torch.tensor(window).permute(2, 0, 1).float().to('cuda')
            
            resize_transform = T.Resize((512, 512))
            window = resize_transform(window.unsqueeze(0)).squeeze(0)
            
            windows.append(window)
            
            start_y += stride
        # print('     number of windows: ', len(windows))
        
        preds = model(windows)
        
        for idx, pred in enumerate(preds):
            # window_timer.start()
            
            filtered_idx = torchvision.ops.nms(pred['boxes'], pred['scores'], nms_thr)
            
            # print("     Time to perform NMS on current window: ", nms_timer.elapsed_time())
            pred_scores = pred['scores'][filtered_idx]
            
            # print('     number of true boxes: ', len(pred_boxes))
            # print('     number of all boxes: ', len(pred['boxes']))
            
            for score in pred_scores:
                if score >= score_thr:
                    total_preds += 1
        # print('Num of preds in current window: ', window_pred_boxes_num)
        start_x += stride
        
    return total_preds

def plot_img_with_boxes(img_fp, pred_boxes, true_boxes):
    img = imread(img_fp)
    for box in pred_boxes:
        x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 10)
    for box in true_boxes:
        x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
        cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 10)
    plt.imshow(img)
    plt.show()
    plt.close()
    return

def check_pred_boxes(pred_boxes, true_boxes, img_tensor, iou_thr):
    true_obj_found = [0] * len(true_boxes)
    
    temp_height, temp_width = img_tensor.shape[1], img_tensor.shape[2]
    true_boxes_map = torch.zeros((temp_height, temp_width))
    true_index_box = torch.zeros((temp_height, temp_width))
    for i, true_box in enumerate(true_boxes):
        x1, y1, x2, y2 = true_box
        x1,y1,x2,y2 = int(x1), int(y1), int(x2), int(y2)
        true_boxes_map[y1:y2, x1:x2] = (y2-y1)*(x2-x1)
        true_index_box[y1:y2, x1:x2] = i + 1
    for pred_box in (pred_boxes):
        x1, y1, x2, y2 = pred_box
        x1,y1,x2,y2 = int(x1), int(y1), int(x2), int(y2)
        intersect_area = (true_boxes_map[y1:y2, x1:x2]>0).sum().item()
        pred_area = (y2 - y1) * (x2 - x1)
        if intersect_area > 0:
            obj_area = torch.max(true_boxes_map[y1:y2, x1:x2]).item()
            if intersect_area >= (obj_area + pred_area - intersect_area) * iou_thr:
                output_index = int(torch.max(true_index_box[y1:y2, x1:x2]).item()) - 1
                true_obj_found[output_index] = 1
    return sum(true_obj_found), len(true_obj_found)

def trainModel(model, parent_dir, train_dataloader,val_dataloader, 
               epochs_num,epoch_begin = 1, nms_thr = 0.1, iou_thr = 0.5, device = 'cuda', chkpt_name = 'model', loss_penalty = 1): 
    recalls = [None] * epochs_num
    
    # Defining the optimizer - adamw
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
    
    # learning rate scheduler - CosineAnnealingLR
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10, eta_min=1e-6)
    
    num_of_epochs = epochs_num
    # Creating a directory for storing models
    model_dir = parent_dir + 'models/'
    if not os.path.exists(model_dir):
        os.mkdir(model_dir)
    
    for epoch in range(epoch_begin, epoch_begin + num_of_epochs, 1):
        model.train()
        print('Training started')
        batch_num = 0
        for batch in train_dataloader: # dataloader here, with collate function
            
            imgs = []
            targets = []
            for item in batch:
                imgs.append(item['image'].clone().detach().to(device))
                targets.append({'boxes':item['target']['boxes'].to(device), 'labels':item['target']['labels'].to(device)})
            
            
            # print('img len:', len(imgs), imgs[0].shape)
            # print('target len:', len(targets))
            result = model(imgs, targets)
            
            batch_num += 1
            # print('result updated')
            # pred - Dict[Tensor], containing classificaiton and regression losses
            # print(batch, '\n')
            # print(result)
            
            bbox_loss = result['bbox_regression']
            # class_loss  = (result['classification'], targets[0]['labels'])
            
            loss = bbox_loss * loss_penalty
            
            loss.backward()
            # Sets the gradients of all optimized tensors to zero
            optimizer.step()
            optimizer.zero_grad()
            
            current_lr = scheduler.get_last_lr()[0]
            if batch_num % 100 == 0 or batch_num == len(train_dataloader):
                
                print("     Epoch[{}/{}],".format(epoch,epochs_num), 
                      "train batch[{}/{}],".format(batch_num, len(train_dataloader)), 
                       "bbox loss: {},".format(bbox_loss), "current learning rate: ", current_lr)
            # print("     Epoch: ", epoch+1, " train batch:", batch_num, " time: ", (total_time)/1000, " bbox loss: ", bbox_loss)
  
        # save model for each epoch
        model_path = model_dir + chkpt_name + '_epoch' + str(epoch) + '.pth'
        torch.save(model.state_dict(), model_path)
        
        # evaluate the model
        
        model.eval()
        print('Validation started')
        true_positive_cnt = 0
        total_cnt = 0
        with torch.no_grad():
            batch_num = 0
            avg_recall = 0
            for batch in val_dataloader: # dataloader here, with collate function
                imgs = []
                targets = []
                for item in batch:
                    imgs.append((item['image']).clone().detach().to(device))
                    targets.append({'boxes':item['target']['boxes'].to(device), 'labels':item['target']['labels'].to(device)})
                        
                preds = model(imgs)      
                batch_num += 1
                for i, img_pred in enumerate(preds):
                    filtered_idx = torchvision.ops.nms(img_pred['boxes'], img_pred['scores'], nms_thr)
                    target = targets[i]
                    temp_img = imgs[i]
                    
                    tp, tot = check_pred_boxes(img_pred['boxes'][filtered_idx], target['boxes'], temp_img, iou_thr)
                    true_positive_cnt += tp
                    total_cnt += tot
                recall = true_positive_cnt / total_cnt
                avg_recall += recall
                if batch_num % 100 == 0 or batch_num == len(val_dataloader):   
                    print("     Epoch[{}/{}],".format(epoch,epochs_num), 
                            " val batch[{}/{}],".format(batch_num, len(val_dataloader)), 
                          "current recall = {}/{} = {}".format(true_positive_cnt, total_cnt, recall))
        # print("     Epoch: ", epoch, " recall: ", recall)
        avg_recall /= batch_num
        recalls[epoch - 1] = avg_recall
        scheduler.step()
        torch.cuda.empty_cache()
        
        x_axis = [i for i in range(epoch_begin, epoch_begin + num_of_epochs)]
        
        # adjust width and height of the plot
        plt.figure(figsize=(15, 5))
        
        plt.xticks(x_axis)
        plt.xlabel('Epoch')
        plt.ylabel('Recall')
        plt.plot(x_axis, recalls, label='Recall', marker='o')
        plt.show()
        plt.tight_layout()
        plt.close('all')
        
    
        
def clean_dir(dir_path):
    for root, dirs, files in os.walk(dir_path):
        for file in files:
            os.remove(os.path.join(root, file))
        for dir in dirs:
            shutil.rmtree(os.path.join(root, dir))
    return


from typing import Tuple
from pandas import DataFrame
from geopandas import GeoDataFrame
from shapely.geometry import Polygon
import numpy as np
import cv2 as cv
import pandas as pd

from utils import imread, im_to_txt_path, imwrite, get_filename, read_label, corners_to_polygon


from os import makedirs
from os.path import isfile

def calculate_iou(box1, box2):
    """
    Calculate the Intersection over Union (IoU) of two bounding boxes.
    
    Parameters:
    box1, box2: Arrays or lists in the format [x_min, y_min, x_max, y_max]
    
    Returns:
    float: IoU value
    """
    x1_min, y1_min, x1_max, y1_max = box1
    x2_min, y2_min, x2_max, y2_max = box2
    
    # Calculate the intersection coordinates
    x_min = max(x1_min, x2_min)
    y_min = max(y1_min, y2_min)
    x_max = min(x1_max, x2_max)
    y_max = min(y1_max, y2_max)
    
    # Calculate the intersection area
    intersection_area = max(0, x_max - x_min) * max(0, y_max - y_min)
    
    # Calculate the area of both boxes
    box1_area = (x1_max - x1_min) * (y1_max - y1_min)
    box2_area = (x2_max - x2_min) * (y2_max - y2_min)
    
    # Calculate the union area
    union_area = box1_area + box2_area - intersection_area
    
    # Calculate the IoU
    iou = intersection_area / union_area
    
    return iou


def clip_value(value, min_value, max_value):
    return max(min(value, max_value), min_value)

# Tile multiple ROIs with labels using paralell processing.
from multiprocessing import Pool
from typing import List, Union, Tuple
from pandas import DataFrame, concat

def random_crop_roi_wrapper(
    roi_fps: List[str], roi_label_dir:str = None, tile_label_dir = None, save_dir: Union[str, List[str]]=None, crop_size: int=512, nproc: int = 10,
    fill: Tuple[int] = (114, 114, 114), box_thr: float = 0.5, 
    notebook: bool = False,
    negative_ratio: float = 1.0, negative_label_ratio: float = 1.0, negatives_only = False, negatives_boxes = None
    
) -> DataFrame:
    """
    
    Args:
        fps: Image filepaths, should be in an '/images/ directory'.
        save_dir: Either a single location to create images and labels dir or
            a list of directories for each filepath passed.
        tile_size: Size of tile, uses square tiles only.
        stride: Stride to use when tiling, if None then it is set equal to 
            tile_size (no overlap between tiles).
        boundary_thr: If ROI has a boundary (for rotated ROIs) then a tile must
            have sufficient area in boundary to be included (0.0 - 1.0).
        nproc: Number of parallel processed to use.
        fill: RGB when padding image.
        box_thr: Area threshold of box that must be in a tile.
        notebook: Select which type of tqdm to use.
       
    Returns:
        Metadata of tiles saved.
        
    """
    if isinstance(save_dir, (list, tuple)):
        if len(save_dir) != len(roi_fps):
            raise Exception('If save_dir is a tuple / list, then it must be the '
                            'same length as the number of filepaths.')
    else:
        save_dir = [save_dir] * len(roi_fps)
    
    if notebook:
        from tqdm.notebook import tqdm
    else:
        from tqdm import tqdm
        
    with Pool(nproc) as pool:
        jobs = [
            pool.apply_async(
                func=random_crop_roi, 
                # roi_fp: str, save_dir: str, crop_size: int, box_area_thr: float = 0.5, fill: Tuple[int] = (114, 114, 114)
                args=(fp, roi_label_dir,tile_label_dir, sd, crop_size, box_thr, fill, 
                      negative_ratio, negative_label_ratio, negatives_only, negatives_boxes)) 

                        
            for fp, sd in zip(roi_fps, save_dir)]
        
        tile_df = [job.get() for job in tqdm(jobs)]
        
    return concat(tile_df, ignore_index=True)


def random_crop_roi(
    roi_fp: str, roi_label_dir = None, tile_label_dir = None, save_dir: str = None, crop_size: int = 512, box_area_thr: float = 0.5, fill: Tuple[int] = (114, 114, 114), 
    negative_ratio: float = 1.0, negative_label_ratio: float = 1.0, negatives_only = False, negatives_boxes = False
) -> DataFrame:
    """
    Randomly crop an ROI image with labels.    
    """
    
    roi_name = roi_fp.split('/')[-1].split('.')[0]
    # create dir
    img_dir = save_dir + 'images/'
    
    if tile_label_dir == None:
        label_dir = save_dir + 'labels/'
    else:
        label_dir = tile_label_dir
    
    makedirs(img_dir, exist_ok=True)
    makedirs(save_dir + 'labels/', exist_ok=True)
    
    # create the dataframe
    tile_df = []
    
    # read the roi image
    orig_img = imread(roi_fp)
    img = orig_img.copy()
    h, w = img.shape[:2]
    
    # print(h,w)
    
    # pad the image to avoid getting tiles not of the right size.
    img = cv.copyMakeBorder(img, 0, crop_size, 0, crop_size, cv.BORDER_CONSTANT, 
                            value=fill)
    
    # look for labels
    if roi_label_dir == None:
        true_label_fp = im_to_txt_path(roi_fp)
    else:
        true_label_fp = roi_label_dir + roi_name + '.txt'
    
    if isfile(true_label_fp):
        # labels = read_label(label_fp, convert=True)
        true_labels = []
        with open(true_label_fp) as label_f:
            for line in label_f.readlines():
                temp = line.split(' ')
                label = temp[0]
                x1, y1, x2, y2 = temp[1:5]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                # print(x1,y1,x2,y2)
        
                true_labels.append([label, x1, y1, x2, y2])

    else:
        true_labels = []
    
    # print('len of true labels: ', len(true_labels))
    
    if not negatives_only:    
        # random crop with labels
        for box in true_labels:
            x_rand = np.random.random()
            y_rand = np.random.random()
            label, box_x1, box_y1, box_x2, box_y2 = box
            
            # print(x1, y1, x2, y2)
            # get the center of the box
            box_width = box_x2 - box_x1
            box_height = box_y2 - box_y1
            # print(center)
            # get the crop size
            # get the top left corner of the crop
            crop_x1 = clip_value(int(box_x1 - (crop_size - box_width) * x_rand), 0, w - crop_size) 
            crop_y1 = clip_value(int(box_y1 - (crop_size - box_height) * y_rand), 0, h - crop_size)
            # get the bottom right corner of the crop
            crop_x2 = int(crop_x1 + crop_size)
            crop_y2 = int(crop_y1 + crop_size)
            # crop the image
            # print(crop_x1, crop_y1, crop_x2, crop_y2)
            crop = img[(crop_y1):(crop_y2), (crop_x1):(crop_x2)]
            # create a name for the crop image / label
            fn = f'{get_filename(roi_fp)}-x{crop_x1}y{crop_y1}.'
            img_fp = img_dir + fn + 'png'
            
            # save the crop image
            imwrite(img_fp, crop)
            # shift the box coordinates to be relative to the crop
            # check if other boxes as well appear in the current crop

            # random crop with labels
            for box in true_labels:
                orig_box_area = (box[3] - box[1]) * (box[4] - box[2])
                type = box[0]
                box_xx1 = clip_value(box[1] - crop_x1,0,crop_size)
                box_yy1 = clip_value(box[2] - crop_y1,0,crop_size)
                box_xx2 = clip_value(box[3] - crop_x1,0,crop_size)
                box_yy2 = clip_value(box[4] - crop_y1,0,crop_size)
                
                # print('orig box: ', box[1], box[2], box[3], box[4])
                # print('shifted box: ', box_xx1, box_yy1, box_xx2, box_yy2)
                
                cropped_box_area = (box_xx2 - box_xx1) * (box_yy2 - box_yy1)
                if cropped_box_area / orig_box_area < box_area_thr:
                    continue
                else:     
                    # save the box as a normalized label in center format
                    box_xc = (box_xx1 + box_xx2) / 2
                    box_yc = (box_yy1 + box_yy2) / 2
                    box_bw = box_xx2 - box_xx1
                    box_bh = box_yy2 - box_yy1
                    label = f'{type} {box_xc / crop_size:.4f} {box_yc / crop_size:.4f} {box_bw / crop_size:.4f} {box_bh / crop_size:.4f}\n'
                    with open(im_to_txt_path(img_fp), 'w') as fh:
                        fh.write(label.strip())
            if isfile(im_to_txt_path(img_fp)):            
                with open(im_to_txt_path(img_fp), mode='r') as file:
                    line_count = sum(1 for line in file)
            tile_df.append([img_fp, roi_fp, crop_x1, crop_y1, crop_x2, crop_y2, crop_size, line_count])
    
    # random crop without labels
    if negatives_boxes == False: # is the boxes negative or postitive annotations? 
        num_positive_crop = len(tile_df)
        num_generated = 0
       
        while num_generated < num_positive_crop * negative_ratio:
            # get the top left corner of the crop
            
            crop_x1 = int(np.random.randint(0, w - crop_size))
            crop_y1 = int(np.random.randint(0, h - crop_size))
            # get the bottom right corner of the crop
            crop_x2 = int(crop_x1 + crop_size)
            crop_y2 = int(crop_y1 + crop_size)
            
            # check if the crop contains any boxes
            crop_contains_boxes = False
            for box in true_labels:
                box_x1, box_y1, box_x2, box_y2 = box[1:5]
                if (box_x1 >= crop_x1 and box_y1 >= crop_y1 and box_x2 <= crop_x2 and box_y2 <= crop_y2):
                    crop_contains_boxes = True
                    break
            if crop_contains_boxes:
                continue
            
            # if the crop contains no labels, then crop the image
            crop = img[(crop_y1):(crop_y2), (crop_x1):(crop_x2)]
            # create a name for the crop image / label
            fn = f'{get_filename(roi_fp)}-x{crop_x1}y{crop_y1}.'
            img_fp = img_dir + fn + 'png'
            # save the crop image
            imwrite(img_fp, crop)
            # save the crop image
            
            tile_df.append([img_fp, roi_fp, crop_x1, crop_y1, crop_x2, crop_y2, crop_size, 0])
            num_generated += 1
    else:
        neg_boxes_dir = label_dir
        neg_boxes = []
        false_pos_label_fp = neg_boxes_dir + roi_name + '.txt'
        # print(false_pos_label_fp)
        
        if isfile(false_pos_label_fp):
            with open(false_pos_label_fp) as neg_label_f:
                for line in neg_label_f.readlines():
                    temp = line.split(' ')
                    neg_boxes.append(temp)
        neg_boxes_df = pd.DataFrame(neg_boxes, columns=['roi_fp','x1', 'y1', 'x2', 'y2','score'])
        neg_boxes_df.sort_values(by='score', ascending=False, inplace=True)
        neg_boxes_sorted = neg_boxes_df.values.tolist()
        neg_boxes = []
        for neg_box in neg_boxes_sorted[0:int(len(neg_boxes_sorted) * negative_label_ratio)]:
            box_x1, box_y1, box_x2, box_y2 = [int(float(x)) for x in neg_box[1:5]]
            neg_boxes.append([box_x1, box_y1, box_x2, box_y2])
        
        # print('len of neg boxes: ', len(neg_boxes))
        # cover all true labels on the roi with certain color
        for box in true_labels:
            x1 = box[1]
            y1 = box[2]
            x2 = box[3]
            y2 = box[4]
            img[y1:y2, x1:x2] = (0,0,0)
            
        for neg_box in neg_boxes:
            x_rand = np.random.random()
            y_rand = np.random.random()
            box_x1, box_y1, box_x2, box_y2 = neg_box
            
            # print(x1, y1, x2, y2)
            # get the center of the box
            box_width = box_x2 - box_x1
            box_height = box_y2 - box_y1
            # print(center)
            # get the crop size
            # get the top left corner of the crop
            crop_x1 = clip_value(int(box_x1 - (crop_size - box_width) * x_rand), 0, w - crop_size) 
            crop_y1 = clip_value(int(box_y1 - (crop_size - box_height) * y_rand), 0, h - crop_size)
            # get the bottom right corner of the crop
            crop_x2 = int(crop_x1 + crop_size)
            crop_y2 = int(crop_y1 + crop_size)
            # crop the image
            # print(crop_x1, crop_y1, crop_x2, crop_y2)
            crop = img[(crop_y1):(crop_y2), (crop_x1):(crop_x2)]
            # create a name for the crop image / label
            fn = f'{get_filename(roi_fp)}-x{crop_x1}y{crop_y1}.'
            img_fp = img_dir + fn + 'png'
            # save the crop image
            # print(img_fp)
            imwrite(img_fp, crop)
            tile_df.append([img_fp, roi_fp, crop_x1, crop_y1, crop_x2, crop_y2, crop_size, 0])
            # shift the box coordinates to be relative to the crop
            # check if other boxes as well appear in the current crop
            
    
    return DataFrame(tile_df, columns=['fp', 'roi_fp', 'x1', 'y1', 'x2', 'y2', 'tile_size', 'num_boxes'])
