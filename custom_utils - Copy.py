import numpy as np
from itertools import repeat
from multiprocessing import Pool
import random
import time
import imageio
import imutils
import cv2

class Video:
    def __init__(self, path):
        self.path = path
        self.container = imageio.get_reader(path, 'ffmpeg')
        self.length = self.container.count_frames()
        self.fps = self.container.get_meta_data()['fps']
    
    def init_head(self):
        self.container.set_image_index(0)
    
    def next_frame(self):
        self.container.get_next_data()
    
    def get(self, key):
        return self.container.get_data(key)
    
    def __call__(self, key):
        return self.get(key)
    
    def __len__(self):
        return self.length

def return_image_data_from_video(video_path, annotation, resize_width=300, frames=[50,100,200]):
    """
    Return images from video. Resize the gray images to 'resize_width' and only take frames specified in 'frames'
    """
    video = Video(video_path)
    objects = []
    for frame in frames:
        if video.__len__() > frame:
            image = video.get(frame)
                
            image_resized = imutils.resize(image, width=resize_width)
            m_height = image.shape[0]/image_resized.shape[0]
            m_width = image.shape[1]/resize_width
            gray = cv2.cvtColor(image_resized, cv2.COLOR_BGR2GRAY)
            
            image_data = {
                'image': image,
                'image_resized': image_resized,
                'm_height': m_height,
                'm_width': m_width,
                'gray': gray,
                'annotation': annotation,
                'video_path': video_path,
                'frame': frame
            }
            objects.append(image_data)
        
    return objects
    
    
def return_image_data_from_video_wrapper(all_videos_batch, videos_per_process, resize_width, frames):
    list_objects = []
    for video_path, annotation in all_videos_batch:
        list_objects.extend(return_image_data_from_video(video_path, annotation, resize_width, frames))
    
    return list_objects
    
def return_image_data_from_video_multi_process(all_videos, list_objects, videos_per_process=25, \
                                                resize_width=300, frames=[50,100,200], processes=4):
    while True:
        if len(list_objects) < videos_per_process * processes * len(frames):
            list_all_videos_batch = []
            for i in range(processes):
                list_all_videos_batch.append(random.choices(all_videos, k=videos_per_process))
            with Pool(processes=processes) as pool:
                    res = pool.starmap(return_image_data_from_video_wrapper, \
                                       zip(list_all_videos_batch, repeat(videos_per_process), \
                                           repeat(resize_width), repeat(frames)))
                    for r in res:
                        list_objects.extend(r)
        else:
            time.sleep(0.001)