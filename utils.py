# General functions / classes
# functions:
# - imwrite
# - imread
# - load_yaml
# - print_opt
# - load_json
# - save_json
# - Timer
# - dict_to_opt
# - save_to_txt
# - im_to_txt_path
# - get_filename
# - create_cases_df
# - create_wsis_df
# - object_metrics
# - read_any_labels
# - draw_boxes
# - get_label_fp
import cv2 as cv
import yaml
from collections import namedtuple
from colorama import Fore, Style
import json
from time import perf_counter
from pandas import DataFrame
import numpy as np
from sklearn.metrics import f1_score, confusion_matrix
import re
from typing import Tuple
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon

from os import remove, makedirs
from os.path import splitext, isfile, join


def imwrite(filepath, im, grayscale=False):
    """Write an image to file using opencv
    
    INPUTS
    ------
    filepath : str
        path to save image to
    im : array-like
        image array
    grayscale : bool
        if True then save image in grayscale, else save as RGB
    
    """
    if grayscale:
        cv.imwrite(filepath, im)
    else:
        cv.imwrite(filepath, cv.cvtColor(im, cv.COLOR_RGB2BGR))
        
        
def imread(impath, grayscale=False):
    """
    Read image from file using opencv, returns as RGB or grayscale.
    
    INPUTS
    ------
    impath : str
        path to image
    grayscale : bool
        if True return the image as grayscale
    
    RETURN
    ------
    numpy array, image
    
    """
    if grayscale:
        return cv.imread(impath, 0)
    else:
        return cv.cvtColor(cv.imread(impath), cv.COLOR_BGR2RGB)

    
def load_yaml(filepath='conf.yaml'):
    """
    Load a yaml file and return the content as a namedtuple.
    
    INPUTS
    ------
    filepath : str
        path to yaml file (default: 'conf.yaml')
    
    RETURN
    ------
    namedtuple
        yaml contents
    
    """
    with open(filepath, 'r') as f:
        cf =  yaml.safe_load(f)

    try:
        return dict_to_opt(cf)
    except ValueError as err:
        print(Fore.RED + f'all keys in \"{filepath}\" must be valid, make sure there is no dashes in the keys' + Style.RESET_ALL)
        raise

        
def print_opt(opt):
    """
    Print out the keys / value of named tuple.
    
    """
    # convert opt to dict
    try:
        opt = opt._asdict()
    except AttributeError:
        opt = vars(opt)
    
    printstr = ''
    for k, v in opt.items():
        printstr += f'{k}={v}, '
        
    if len(printstr):
        printstr = printstr[:-2]
        print('   opt:\t', end='')
        print(printstr)
        print()
    else:
        print('  opt is empty\n')

        
def load_json(path):
    """Load a json file.
    
    """
    with open(path, 'r') as fh:
        json_content = json.load(fh)
    return json_content


def save_json(path, data):
    """Save a variable to json file
    
    param: path is the path to save json file
    param: data, the variable to jsonify
    
    """
    with open(path, 'w') as fh:
        json.dump(data, fh)
        
        
class Timer():
    def __init__(self, fp):
        """Timer class."""
        self.fp = fp
        
        # start the document
        if not isfile(self.fp):
            with open(self.fp, 'w') as fh:
                fh.write('')
                
        self.running = False
        self.time = None
        
    def start(self):
        """Start the timer."""
        if not self.running:
            self.running = True
            self.time = perf_counter()
            
    def stop(self):
        """Stop the timer."""
        if self.running:
            self.running = False
            
            delta = perf_counter() - self.time
            
            with open(self.fp, 'a') as fh:
                fh.write(f'{delta:.0f}\n')
            
            self.time = None
        else:
            print(Fore.YELLOW, Style.BRIGHT, 
                  'No start timer, run .start() first.', Style.RESET_ALL)

            
def dict_to_opt(opt_dict):
    """Convert a dictionary into an opt style variable, with dot format to access fields"""
    # replace any keys with dashes to underscores
    valid_dict = {}
    
    for k, v in opt_dict.items():
        if isinstance(k, str):
            valid_dict[k.replace('-', '_')] = v
        else:
            valid_dict[k] = v
    
    return namedtuple("ObjectName", valid_dict.keys())(*valid_dict.values())


def save_to_txt(save_filepath, string):
    """Save a string to a text file"""
    with open(save_filepath, 'w') as fh:
        fh.writelines(string)
        
        
def im_to_txt_path(impath: str, txt_dir: str = 'labels'):
    """Replace the last occurance of /images/ to /labels/ in the given image path and change extension to .txt"""
    splits = impath.rsplit('/images/', 1)
    return splitext(f'/{txt_dir}/'.join(splits))[0] + '.txt'

def txt_to_im_path(impath: str, txt_dir: str = 'images'):
    """Replace the last occurance of /labels/ to /images/ in the given image path and change extension to .png"""
    splits = impath.rsplit('/labels/', 1)
    return splitext(f'/{txt_dir}/'.join(splits))[0] + '.png'


def get_filename(path: str, prune_ext: bool = True, replaces_spaces: bool = False) -> str:
    """Get filename from a file path.
    
    Args:
        path: Filepath to get file name from.
        prune_ext: If True then the filename is returned without extension.
        replaces_spaces: If True then space characters are replaced with underscores, multiple sequential spaces are 
            replaced by a single underscore.
        
    Returns:
        Filename.
        
    """
    filename = path.split('/')[-1]
    
    if prune_ext:
        filename = splitext(filename)[0]
    
    if replaces_spaces:
        filename = re.sub(' +', '_', filename)
    
    return filename


def create_cases_df(annotations, cases_cols=None):
    """Create cases dataframe
    
    INPUTS
    ------
    annotations : dict
        keys are item ids, values are the items
    cases_cols : list (default: None)
        list of keys in meta to add to the dataframe as columns, if None then default is a list of columns
    
    RETURNS
    -------
    : dataframe
        metadata dataframe for the cases
    parent_id_map : dict
        map of the item name to ids
        
    """
    # loop through the inference cohort to get case information for the annotation cohort
    cases_df = []
    
    parent_id_map = {}  # map WSI name to the id of WSI / item in the inference cohort
    
    if cases_cols is None:
        cases_cols = [
            'Braak_stage', 'ABC', 'age_at_death', 'Clinical_Dx', 'Other_NP_Dx_AD', 'Other_NP_Dx_LBD', 'Other_NP_Dx_Misc_1', 'Other_NP_Dx_Misc_2',
            'Other_NP_Dx_Misc_3', 'Other_NP_Dx_Misc_4', 'Other_NP_Dx_TAU', 'Other_NP_Dx_TDP', 'Other_NP_Dx_Vascular', 'Primary_NP_Dx', 'race', 
            'region', 'sex', 'Thal'
        ]
    
    cases_added = []  # only add one entry per case
    
    for wsi_id, item in annotations.items():
        wsi_name = item['name']
        meta = item['meta'] if 'meta' in item else {}
        
        if 'Braak_stage' not in meta and 'Braak Stage' in meta:
            meta['Braak_stage'] = meta['Braak Stage']
            
        if 'Clinical_Dx' not in meta and 'Clinical Dx' in meta:
            meta['Clinical_Dx'] = meta['Clinical Dx']
            
        if 'Primary_NP_Dx' not in meta and 'Primary NP Dx' in meta:
            meta['Primary_NP_Dx'] = meta['Primary NP Dx']
            
        if 'age_at_death' not in meta and 'age at death' in meta:
            meta['age_at_death'] = meta['age at death']
        
        # map this WSI mage to its id
        parent_id_map[wsi_name] = wsi_id
        
        if meta['case'] not in cases_added:
            # add this case
            cases_added.append(meta['case'])
            row = [meta['case']]
            
            for case_col in cases_cols:
                row.append(meta[case_col] if case_col in meta else '')
            
            cases_df.append(row)
           
    # return as a dataframe, also return the map of the image name to parent id
    return DataFrame(data=cases_df, columns=['case'] + cases_cols), parent_id_map


def create_wsis_df(annotations, parent_id_map=None, meta_keys=None):
    """Create wsis dataframe
    
    INPUTS
    ------
    annotations : dict
        keys are item ids, values are the items. The items should have keys: name, meta, and scan_mag, cohort
    parent_id_map : dict (default: None)
        pass a parent id dict, mapping names to item ids. If None then it will be empty strings
    meta_keys : list (default: None)
        list of meta keys, default is None 
        
    RETURN
    ------
    : dataframe
        WSI dataframe data
    
    """
    if parent_id_map is None:
        parent_id_map = {}
        
    if meta_keys is None:
        meta_keys = []
        
    wsis_df = []
    
    # loop through all items in the annotated cohort
    for item_id, item in annotations.items():
        # add the info of this WSI
        wsi_name = item['name']
        meta = item['meta']
        
        parent_id = parent_id_map[wsi_name] if wsi_name in parent_id_map else ''
        
        row = [item['name'], item_id, parent_id] + [item[k] for k in ('scan_mag', 'cohort')]
        
        meta = item['meta'] if 'meta' in item else {}
        
        for k in meta_keys:
            row.append(meta[k] if k in meta else '')
        
        wsis_df.append(row)
        
    # return as a Dataframe
    return DataFrame(data=wsis_df, columns=['wsi_name', 'wsi_id', 'parent_id', 'scan_mag', 'cohort'] + meta_keys)


def object_metrics(
    matches: DataFrame, labels: list = None, bg_label: int = None) -> list[np.array, float, float, np.array, dict
    ]:
    """Given an dataframe of true and pred matches, output from match_labels(), calculate metrics. Mainly report
    the per-class F1 score, micro & macro F1 scores, average IoU score for each class when correctly predicting, 
    and the confusion matrix.
    
    Args:
        matches: Contains columns: true, pred, iou, x1, y1, x2, y2, px1, py1, px2, py2, conf.
    """
    # calculate the F1 scores
    true = matches['true'].tolist()
    preds = matches['pred'].tolist()
    
    if labels is None:
        labels = list(range(np.max(true + preds) + 1))
        
    if bg_label is None:
        bg_label = []
    else:
        bg_label = [bg_label]
        
    per_class_f1 = f1_score(true, preds, average=None, labels=labels)
    micro_f1 = f1_score(true, preds, average='micro', labels=labels)
    macro_f1 = f1_score(true, preds, average='macro', labels=labels)
    cm = confusion_matrix(true, preds, normalize='true', labels=labels + bg_label).T
    
    ious = {}
    
    for _, r in matches.iterrows():
        if r['true'] in labels and r['pred'] in labels and r['true'] == r['pred']:
            if r['true'] not in ious:
                ious[r['true']] = []
            ious[r['true']].append(r.iou)
            
    for k in list(ious.keys()):
        ious[k] = np.mean(ious[k])
        
    return per_class_f1, micro_f1, macro_f1, cm, ious


def delete_file(fp: str):
    """Delete a file and make sure it is deleted before continuing.
    
    Args:
        fp: Filepath.
        
    """
    while isfile(fp):
        try:
            remove(fp)
        except OSError:
            pass

        
def read_any_labels(fp: str, im_shape: Tuple[int, int]) -> np.array:
    """Read any label in either format and return it in non-YOLO format.
    
    Args:
        fp: Label filepath.
        im_shape: width, height of image.
        
    Returns:
        Boxes in non-yolo format (label, x1, y1, x2, y2, conf optional).
    
    """
    boxes = []
    
    if isfile(fp):
        with open(fp, 'r') as fh:
            for line in fh.readlines():
                line = [float(l) for l in line.strip().split(' ')]

                if max(line[1:5]) > 1:
                    # Already in non-yolo format
                    boxes.append(line)
                else:
                    xc, yc, bw, bh = line[1:5]
                    
                    bw2 = bw / 2
                    bh2 = bh / 2
                    
                    x1, y1, x2, y2 = xc - bw2, yc - bh2, xc + bw2, yc + bh2
                    line[1] = int(x1 * im_shape[0])
                    line[2] = int(y1 * im_shape[1])
                    line[3] = int(x2 * im_shape[0])
                    line[4] = int(y2 * im_shape[1])
                    
                    boxes.append(line)
                    
    return np.array(boxes)


def draw_boxes(img: np.array, boxes: np.array, lw: int = 10) -> np.array:
    """Draw NFT boxes on an image.
    
    Args:
        img: Image.
        boxes: Boxes in non-yolo format (label, x1, y1, x2, y2, conf optional).
        lw: Line with when drawing boxes.
        
    Returns:
        Image with drawn boxes.
        
    """
    img = img.copy()
    
    for box in boxes:
        label, x1, y1, x2, y2 = box[:5].astype(int)
        
        img = cv.rectangle(img, (x1, y1), (x2, y2), 
                           (255, 0, 0) if label else (0, 0, 255), lw)
        
    return img


def get_label_fp(fp: str, label_dir: str = None) -> str:
    """Get the text label file for an image.
    
    Args:
        fp: Filepath to image.
        label_dir: If passed then this is the directory wher the label file is,
            otherwise it is grabbed parallel to the filepath in labels dir.
            
    Returns:
        Label text filepath.
    
    """
    if label_dir is None:
        return im_to_txt_path(fp)
    else:
        return join(label_dir, get_filename(fp) + '.txt')
    
def read_label(filepath, im_shape=None, shift=None, convert=False):
    """Read a yolo label text file. It may contain a confidence value for the labels or not, will handle both cases
    
    INPUTS
    ------
    filepath : str
        the path of the text file
    im_shape : tuple or int (default: None)
        image width and height corresponding to the label, if an int it is assumed both are the same. Will scale coordinates
        to int values instead of normalized if given
    shift : tuple or int (default: None)
        shift value in the x and y direction, if int it is assumed to be the same in both. These values will be subtracted and applied
        after scaling if needed
    convert : bool (default: False)
        If True, convert the output boxes from yolo format (label, x-center, y-center, width, height, conf) to (label, x1, y1, x2, y2, conf)
        where point 1 is the top left corner of box and point 2 is the bottom corner of box
    
    RETURN
    ------
    coords : array
        coordinates array, [N, 4 or 5] depending if confidence was in input file
    
    """
    coords = []
    
    with open(filepath, 'r') as fh:
        for line in fh.readlines():
            if len(line):
                coords.append([float(ln) for ln in line.strip().split(' ')])
                
    coords = np.array(coords)
    
    # scale coords if needed
    if im_shape is not None:
        if isinstance(im_shape, int):
            w, h = im_shape, im_shape
        else:
            w, h = im_shape[:2]
            
        coords[:, 1] *= w
        coords[:, 3] *= w
        coords[:, 2] *= h
        coords[:, 4] *= h
        
    # shift coords
    if shift is not None:
        if isinstance(shift, int):
            x_shift, y_shift = shift, shift
        else:
            x_shift, y_shift = shift[:2]
            
        coords[:, 1] -= x_shift
        coords[:, 2] -= y_shift
        
    if convert:
        coords[:, 1:5] = convert_box_type(coords[:, 1:5])
        
    return coords


def get_contours(mask, enclosed_contours=False, min_points=3):
    """Get object contours from a binary mask using the OpenCV method.

    INPUTS
    ------
    mask : array-like
        binary mask to extract contours from - note that it must be in np.uint8 dtype, bool dtype won't work and this
        function will try to convert it to numpy uint8 form
    enclosed_contours : bool (default: False)
        if True then contours enclosed in other contours will be returned, otherwise they will be filtered
    min_points : int (default: 3)
        the minimum number of points a contour must have to be included

    RETURN
    ------
    contours : list
        list of object contours

    """
    # convert to uint8 dtype if needed
    if mask.dtype == 'bool':
        mask = mask.astype(np.bool)

    # extract contours - note that a default method is used for extracting contours
    contours, h = cv.findContours(mask, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

    # filter out contours that don't have enough points
    temp_contours = []
    for contour in contours:
        if contour.shape[0] >= min_points:
            temp_contours.append(contour)
    contours = temp_contours

    # filter out enclosed contours if needed
    if not enclosed_contours:
        temp_contours = []
        for i, contour in enumerate(contours):
            if h[0, i, 3] == -1:
                temp_contours.append(contour)
        contours = temp_contours

    return contours


def contour_to_line(contour):
    """convert a contour in opencv format to a string of x y points (i.e. x1 y1 x2 y2 x3 y3 ... xn yn)"""
    return ' '.join(contour.flatten().astype(str))


def xys_to_line(xys, shift=None):
    """Convert an array of xy coordiantes to a string of xy coordinates with spaces
    
    INPUTS
    ------
    xys : array-like
        shape (N, 2) with N being the number of points and each row having x, y coordinates
    shift : tuple / list (default: None)
        lenght of 2, corresponding to an x and y shift (subtracted) to the xys array
        
    RETURN
    ------
    : str
        the coordiantes flattened in a string as x1 y1 x2 y2 x3 y3 ... xN yN
        
    """
    if shift is not None:
        if len(shift) != 2:
            raise Exception('shift parameter must be of lenght 2')
            
        xys = (xys - shift).astype(int)
    
    return ' '.join([str(v) for v in xys.flatten()])


def line_to_xys(line, shift=None):
    """Convert a line of x y coordinates separated by spaces.
    
    INPUTS
    ------
    line : str
        x y coordinates for points
    shift : tuple / list (default: None)
        lenght of 2, shift the coordinates (subtracted) to the xys array
        
    RETURN
    ------
    xys : array-like
        shape (N, 2) with N being the number of points and each row having x, y coordinates
        
    """
    xys = np.reshape(np.array([int(c) for c in line.split(' ')]), (-1, 2))
    
    if shift is not None:
        if len(shift) != 2:
            raise Exception('')
            
        xys = (xys - shift).astype(int)
        
    return xys


def pt_in_polygon(pt_x, pt_y, vertices):
    """Provide the x, y coordinates of a point and the vertices in a polygon and return True if the point
    is in the polygon or False otherwise.
    
    INPUTS
    ------
    pt_x, pt_y : int or float
        x, y coordinates if the point
    vertices : list
        list of x, y tuples, the vertices of the polygon
        
    RETURN
    ------
    True or False
    
    """
    point = Point(pt_x, pt_y)
    polygon = Polygon(vertices)
    
    return polygon.contains(point)


def tile_im_with_boxes(im, boxes, save_dir, savename='', rotated_mask=None, box_thr=0.45, tile_size=1280, stride=960, 
                       pad_rgb=(114, 114, 114), rotated_thr=0.2, ignore_existing=False):
    """Tile an image containing box annotations. The image should be rectangular, but rotated images are supported. In the case of 
    rotated images a mask may be passed to identify the region of the image that is the rectangular region of interest. All other regions
    will be ignored.
    
    INPUTS
    ------
    im : array-like
        and RGB or gray-scale image that will be tiled, or broken into smaller regions
    boxes : list
        list of annotations boxes in image, each box has format [label, x1, y1, x2, y2] where point 1 is the top left corner of the box
        and point 2 is the bottom right corner of box. All coordinates are relative to (0, 0) being the top left corner of the image
    save_dir : str
        dir to create image and labels directories to save the images and label text file combos for the tiles
    savename : str (default: '')
        each tile image and label text file will include the coordinates of the tile, but will be prepended by the value of this 
        parameter, by default the prepend is an empty string
    rotated_mask : array-like (default: None)
        mask to specify region of interest inside image, for rotated images only (1: inside ROI, 0: outside of ROI)
    box_thr : float (default: 0.45)
        percentage of box (by area) that must be in tile to be included (from its original size)
    tile_size : int (default: 1280)
        size of tiles
    stride : int (default: 960)
        amount distance to travel between adjacent tiles, if it is less than tile_size then there will be overlap between tiles
    pad_rgb : tuple (default: (114, 114, 114))
        RGB to use when padding the image
    rotated_thr : float (default: 0.2)
        for rotated images, percentage of tile area that must be in rotated region to be included
    ignore_existing : bool (default: False)
        if True, then images will not save if a file of the name already exists
        
    RETURN
    ------
    save_paths : list
        list of image file paths that were saved, also includes x, y coordinate for that image
        
    """
    # create save locations
    im_dir = join(save_dir, 'images')
    label_dir = join(save_dir, 'labels')
    
    makedirs(im_dir, exist_ok=True)
    makedirs(label_dir, exist_ok=True)
    
    # create a mask for the boxes, with each box having a unique label
    height, width = im.shape[:2]
    boxes_mask = np.zeros((height, width), dtype=np.uint8)
    
    # for each box, track its area
    box_area_thrs = {}
    boxes_labels = {}
    
    for i, box in enumerate(boxes):
        x1, y1, x2, y2 = box[1:]
        x1 = int(x1)
        y1 = int(y1)
        x2 = int(x2)
        y2 = int(y2)
        box_area_thrs[i+1] = (x2 - x1) * (y2 - y1) * box_thr
        boxes_labels[i+1] = box[0]
        boxes_mask = cv.drawContours(boxes_mask, np.array([[(x1, y1), (x2, y1), (x2, y2), (x1, y2)]]), -1, i+1, cv.FILLED)
        
    # pad the edges of image
    im = cv.copyMakeBorder(im, 0, tile_size, 0, tile_size, cv.BORDER_CONSTANT, value=pad_rgb)

    save_paths = []
    
    # calculate the area of a tile in rotated image to include
    area_thr = tile_size ** 2 * rotated_thr
    
    for x in range(0, width, stride):
        for y in range(0, height, stride):
            # ignore this tile if it is not sufficiently in rotated image
            if rotated_mask is not None and np.count_nonzero(rotated_mask[y:y+tile_size, x:x+tile_size]) < area_thr:
                continue
                
            filename = f'{savename}x{x}y{y}.'  # filename
            im_path = join(im_dir, filename + 'png')
                
            # grab tile
            tile_mask = boxes_mask[y:y+tile_size, x:x+tile_size].copy()
            tile_im = im[y:y+tile_size, x:x+tile_size, :].copy()

            # track lines to write to file for labels
            lines = ''

            # check each unique int label in tile
            for i in np.unique(tile_mask):
                if i > 0:
                    # ignore this object if not enough in tile
                    if np.count_nonzero(tile_mask == i) > box_area_thrs[i]:
                        # add the line for this object: class x_center y_center width height
                        # # normalize all values to 0 - 1 by dividing the tile size
                        yxs = np.where(tile_mask == i)

                        ymin, xmin = np.min(yxs, axis=1)
                        ymax, xmax = np.max(yxs, axis=1)

                        xcenter, ycenter = (xmax + xmin) / 2 / tile_size, (ymax + ymin) / 2 / tile_size
                        box_w, box_h = (xmax - xmin) / tile_size, (ymax - ymin) / tile_size

                        lines += f'{boxes_labels[i]} {xcenter} {ycenter} {box_w} {box_h}\n'

            # save image and label text file
            save_paths.append([im_path, x, y])  # append save name and coords to return
            
            # save the image only if it does not exist, or ignore_existing is False
            if not isfile(im_path) or not ignore_existing:
                imwrite(im_path, tile_im)
            
            # save its label text file if there are labels for tile
            if len(lines):
                save_to_txt(join(label_dir, filename + 'txt'), lines)
    
    return save_paths


def convert_box_type(box):
    """Convert a box type from YOLO format (x-center, y-center, box-width, box-height) to (x1, y1, x2, y2) where point 1 is the
    top left corner of box and point 2 is the bottom right corner
    
    INPUT
    -----
    box : array
        [N, 4], each row a point and the format being (x-center, y-center, box-width, box-height)
        
    RETURN
    ------
    new_box : array
        [N, 4] each row a point and the format x1, y1, x2, y2
        
    """
    # get half the box height and width
    half_bw = box[:, 2] / 2
    half_bh = box[:, 3] / 2
    
    new_box = np.zeros(box.shape, dtype=box.dtype)
    new_box[:, 0] = box[:, 0] - half_bw
    new_box[:, 1] = box[:, 1] - half_bh
    new_box[:, 2] = box[:, 0] + half_bw
    new_box[:, 3] = box[:, 1] + half_bh
    
    return new_box


def contours_to_points(contours):
    """Convert a list of opencv contours (i.e. contour shape is (num_points, 1, 2) with x, y order) to a list of x,y point in format ready
    to push as DSA annotations. This form is a list of lists with [x, y, z] format where the z is always 0
    
    INPUTS
    ------
    contours : list
        list of numpy arrays in opencv contour format
        
    """
    points = []
    
    for contour in contours:
        
        points.append([[float(pt[0][0]), float(pt[0][1]), 0] for pt in contour])
        
    return points
        
    
def corners_to_polygon(x1: int, y1: int, x2: int, y2: int) -> Polygon:
    """Return a Polygon from shapely with the box coordinates given the top left and bottom right corners of a 
    rectangle (can be rotated).
    
    Args:
        x1, y1, x2, y2: Coordinates of the top left corner (point 1) and the bottom right corner (point 2) of a box.
        
    Returns:
        Shapely polygon object of the box.
        
    """
    return Polygon([(x1, y1), (x2, y1), (x2, y2), (x1, y2)])

def center_to_polygon(xc: int, yc: int, w: int, h: int) -> Polygon:
    """Return a Polygon from shapely with the box coordinates given the center x, y coordinates, with width and height
    
    Args:
        xc, yc, w, h
        
    Returns:
        Shapely polygon object of the box.
        
    """
    x1 = int(xc - w/2)
    x2 = int(xc + w/2)
    y1 = int(yc - h/2)
    y2 = int(yc + h/2)
    
    return Polygon([(x1, y1), (x2, y1), (x2, y2), (x1, y2)])


