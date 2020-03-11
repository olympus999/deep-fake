import numpy as np
from itertools import repeat
from multiprocessing import Pool
from yolo3.model import preprocess_true_boxes
from yolo3.utils import get_random_data
import gc
import time

def data_generator(annotation_lines, batch_size, input_shape, anchors, num_classes, random):
    '''data generator for fit_generator'''
    image_data = []
    box_data = []
    for i in range(len(annotation_lines)):
        image, box = get_random_data(annotation_lines[i], input_shape, random=random)
        image_data.append(image)
        box_data.append(box)
    image_data = np.array(image_data)
    box_data = np.array(box_data)
    y_true = preprocess_true_boxes(box_data, input_shape, anchors, num_classes)
    return [image_data, *y_true], np.zeros(batch_size)
    
def data_generator_multi_process(annotation_lines, batch_size, input_shape, anchors, num_classes, arr, random, processes=12):
    idx = 0
    max_idx = len(annotation_lines)
    while True:
        if len(arr) < 60:
            batch_annotation_lines = []
            for i in range(processes):
                start_idx = idx * batch_size
                end_idx = start_idx + batch_size
                if start_idx >= max_idx:
                    idx = 0
                    start_idx = idx*batch_size
                    end_idx = start_idx + batch_size
                batch_annotation_lines.append(annotation_lines[start_idx:end_idx])
            with Pool(processes=processes) as pool:
                res = pool.starmap(data_generator, \
                                   zip(batch_annotation_lines, repeat(batch_size), repeat(input_shape), \
                                       repeat(anchors), repeat(num_classes), repeat(random)))
                for X, Y in res:
                    arr.append((X, Y))
                gc.collect()
        else:
            time.sleep(0.001)