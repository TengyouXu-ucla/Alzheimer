from typing import Tuple
from pandas import DataFrame
from geopandas import GeoDataFrame
from shapely.geometry import Polygon
import numpy as np
import cv2 as cv

from utils import imread, im_to_txt_path, imwrite, get_filename, read_label, corners_to_polygon

from os import makedirs
from os.path import isfile


def tile_roi_with_labels(
    fp: str, save_dir: str, tile_size, stride: int = None, 
    boundary_thr: float = 0.2, fill: Tuple[int] = (114, 114, 114), 
    box_thr: float = 0.5
) -> DataFrame:
    """Tile an ROI image with labels.
    
    Args:
        fp: Image filepath, should be in an '/images/ directory'.
        save_dir: Location to save images and labels.
        tile_size: Size of tile, uses square tiles only.
        stride: Stride to use when tiling, if None then it is set equal to 
            tile_size (no overlap between tiles).
        boundary_thr: If ROI has a boundary (for rotated ROIs) then a tile must
            have sufficient area in boundary to be included (0.0 - 1.0).
        fill: RGB when padding image.
        box_thr: Area threshold of box that must be in a tile.
       
    Returns:
        Metadata of tiles saved.
        
    """
    # read the image
    img = imread(fp)
    h, w = img.shape[:2]
    
    # look for labels and boundaries
    label_fp = im_to_txt_path(fp)
    boundary_fp = im_to_txt_path(fp, txt_dir='boundaries')
    
    if isfile(label_fp):
        # labels = read_label(label_fp, convert=True)
        labels = []
        with open(label_fp) as label_f:
            for line in label_f.readlines():
                temp = line.split(' ')
                labels.append(temp)
    else:
        labels = []
    
    # Convert the labels into a GeoDataFrame.
    label_df = []
    
    for box in labels:
        label = box[0]
        x1, y1, x2, y2 = box[1:5]
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        # print(x1,y1,x2,y2)
        
        label_df.append([label, x1, y1, x2, y2, 
                         corners_to_polygon(x1, y1, x2, y2)])
        
    label_df = GeoDataFrame(
        label_df, 
        columns=['label', 'x1', 'y1', 'x2', 'y2', 'geometry']
    )
    
    # print(label_df)
    
    label_areas = label_df.area
    
    # For the boundary, create a polygon object.
    if isfile(boundary_fp):
        # format the boundaries in to a countour shape
        with open(boundary_fp, 'r') as fh:
            boundaries = [float(c) for c in fh.readlines()[0].split(' ')]
            
        # scale to the image size.
        roi = Polygon(
            (np.reshape(boundaries, (-1, 2)) * [w, h]).astype(int)
        )
    else:
        # Default: the whole image is the ROI.
        roi = corners_to_polygon(0, 0, w, h)  
    
    img_dir = save_dir + 'images/'
    label_dir = save_dir + 'labels/'
    makedirs(img_dir, exist_ok=True)
    makedirs(label_dir, exist_ok=True)
    
    # Default stride is no-overlap.
    if stride is None:
        stride = tile_size  # default behavior - no overlap
    
    # Pad the image to avoid getting tiles not of the right size.
    img = cv.copyMakeBorder(img, 0, tile_size, 0, tile_size, cv.BORDER_CONSTANT, 
                            value=fill)
    
    # Get the top left corner of each tile.
    xys = list(((x, y) for x in range(0, w, stride) 
                for y in range(0, h, stride)))
        
    tile_df = []  # track tile data.

    # Pre-calculate the number of pixels in tile that must be in ROI to include.
    intersection_thr = tile_size**2 * boundary_thr
    
    # loop through each tile coordinate
    for xy in xys:
        # Check if this tile is sufficiently in the boundary.
        x1, y1 = xy
        x2, y2 = x1 + tile_size, y1 + tile_size
        
        tile_pol = corners_to_polygon(x1, y1, x2, y2)
        intersection = roi.intersection(tile_pol).area
        
        if intersection > intersection_thr:
            # Get the tile image.
            tile = img[y1:y2, x1:x2]
            
            # Create a name for the tile image / label.
            fn = f'{get_filename(fp)}-x{x1}y{y1}.'
            
            img_fp = img_dir + fn + 'png'
            
            tile_df.append([img_fp, fp, x1, y1, tile_size])
            
            if not isfile(img_fp):
                imwrite(img_fp, tile)
                
            # Find all boxes that intersect
            label_intersection = label_df.geometry.intersection(tile_pol).area
            # print('label intersection' , label_intersection)
            
            tile_boxes = label_df[label_intersection / label_areas > box_thr]
            
            # save these as normalized labels, threshold the box edges.
            if len(tile_boxes):
                labels = ''
                
                for _, r in tile_boxes.iterrows():
                    # Shift coordinates to be relative to this tile.
                    
                    # print(r.x1, r.y1, r.x2, r.y2)
                    
                    xx1 = np.clip(r.x1 - x1, 0, tile_size) 
                    yy1 = np.clip(r.y1 - y1, 0, tile_size) 
                    xx2 = np.clip(r.x2 - x1, 0, tile_size) 
                    yy2 = np.clip(r.y2 - y1, 0, tile_size) 

                    xc, yc = (xx1 + xx2) / 2, (yy1 + yy2) / 2
                    bw, bh = xx2 - xx1, yy2 - yy1
                    
                    xc_norm, yc_norm, w_norm, h_norm = xc / tile_size, yc / tile_size, bw / tile_size, bh / tile_size
                    labels += f'{r.label} {xc_norm:.4f} {yc_norm:.4f} {w_norm:.4f} {h_norm:.4f}\n'
                    # print(bw, bh)
                    
                # print('labels fp: ', im_to_txt_path(img_fp))
                with open(im_to_txt_path(img_fp), 'w') as fh:
                    fh.write(labels.strip())
                    
    return DataFrame(tile_df, columns=['fp', 'roi_fp', 'x', 'y', 'tile_size'])



def clip_value(value, min_value, max_value):
    return max(min(value, max_value), min_value)
def random_crop_roi(
    roi_fp: str, save_dir: str, crop_size: int, box_area_thr: float = 0.5, fill: Tuple[int] = (114, 114, 114), negative_ratio: float = 1.0
) -> DataFrame:
    """
    Randomly crop an ROI image with labels.    
    """
    
    # create dir
    img_dir = save_dir + 'images/'
    label_dir = save_dir + 'labels/'
    makedirs(img_dir, exist_ok=True)
    makedirs(label_dir, exist_ok=True)
    
    # create the dataframe
    tile_df = []
    
    # read the roi image
    img = imread(roi_fp)
    h, w = img.shape[:2]
    
    # pad the image to avoid getting tiles not of the right size.
    img = cv.copyMakeBorder(img, 0, crop_size, 0, crop_size, cv.BORDER_CONSTANT, 
                            value=fill)
    
    # look for labels
    label_fp = im_to_txt_path(roi_fp)
    
    if isfile(label_fp):
        # labels = read_label(label_fp, convert=True)
        labels = []
        with open(label_fp) as label_f:
            for line in label_f.readlines():
                temp = line.split(' ')
                labels.append(temp)
    else:
        labels = []
    
    label_df = []
    
    for box in labels:
        label = box[0]
        x1, y1, x2, y2 = box[1:5]
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        # print(x1,y1,x2,y2)
        
        label_df.append([label, x1, y1, x2, y2])
    
    # random crop with labels
    for box in label_df:
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
        print(crop_x1, crop_y1, crop_x2, crop_y2)
        crop = img[(crop_y1):(crop_y2), (crop_x1):(crop_x2)]
        # create a name for the crop image / label
        fn = f'{get_filename(roi_fp)}-x{crop_x1}y{crop_y1}.'
        img_fp = img_dir + fn + 'png'
        
        # save the crop image
        imwrite(img_fp, crop)
        # shift the box coordinates to be relative to the crop
        # check if other boxes as well appear in the current crop

        # random crop with labels
        for box in label_df:
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
        for box in label_df:
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
    
    return DataFrame(tile_df, columns=['fp', 'roi_fp', 'x1', 'y1', 'x2', 'y2', 'tile_size', 'num_boxes'])