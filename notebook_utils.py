import os
import sys
py_dll_path = os.path.join(sys.exec_prefix, 'Library', 'bin')
os.add_dll_directory(py_dll_path)
import traceback
import numpy as np
import multiprocessing as mp
import time
import cv2
import random
import queue
import imutils
from imutils.video import FileVideoStream
from multiprocessing import Queue
from multiprocessing import shared_memory
from PIL import Image
from facenet_pytorch import MTCNN
device = 'cuda'

def resize_and_pad_faces(crops, q_log):
    args = {}
    new_shape = (224,224)
    resized_crops = []
    for crop in crops:
        img = crop
        if np.argmax(img.shape[:2]) == 0:
            args['height'] = new_shape[0]
        else:
            args['width'] = new_shape[1]
        try:
            resized_img = imutils.resize(img, **args)
#             if augment:
#                 resized_img = self.augment_img(resized_img)

            if np.argmax(img.shape[:2]) == 0:
                diff = new_shape[1] - resized_img.shape[1]
                resized_img = np.pad(resized_img, ((0,0), (0,diff), (0,0)), 'constant', constant_values=0)
            else:
                diff = new_shape[0] - resized_img.shape[0]
                resized_img = np.pad(resized_img, ((0,diff), (0,0), (0,0)), 'constant', constant_values=0)
            resized_crops.append(resized_img)

        except:
            tb = traceback.format_exc()
            q_log.put((tb))

    return resized_crops

def crop_faces(frames, boxes, resize):
    crops = []
    for idx, frame in enumerate(frames):
        if boxes[idx] is not None:
            box = boxes[idx][0]
            if len(boxes[idx]) > 1 or len(boxes[idx]) == 0:
                raise(MoreThanOneFaceError('More than one face error'))
            crop = np.array(frames[idx].crop(box/resize))
            crops.append(crop)
    
    return crops

def return_resized_crops(frames_pil, boxes, resize, q_log):
    try:
        crops = crop_faces(frames_pil, boxes, resize)
        crops_numpy = [np.array(x) for x in crops]
    except MoreThanOneFaceError:
        return False
    else:
        resized_crops = resize_and_pad_faces(crops_numpy, q_log)
        return resized_crops
    
def get_numpy_from_memory(memory_frames, memory_shape):
    existing_shm = shared_memory.SharedMemory(name=memory_frames['name'])
    correct_memory_shape = (memory_shape[0], \
                            memory_frames['shape'][1], \
                            memory_frames['shape'][2], \
                            3)
    frames_np = np.ndarray(correct_memory_shape, \
                           dtype=memory_frames['dtype'], \
                           buffer=existing_shm.buf)
    frames_np = np.copy(frames_np[memory_frames['loc_start']:memory_frames['loc_end']])
    existing_shm.close()
    return frames_np
    
def put_numpy_array_to_memory(memory_name, \
                              memory_shape, \
                              numpy_array, \
                              memory_idx_start, \
                              memory_idx_end):
    existing_shm = shared_memory.SharedMemory(name=memory_name)
    correct_memory_shape = (memory_shape[0], \
                            numpy_array.shape[1], \
                            numpy_array.shape[2], \
                            3)
    frames_np = np.ndarray(correct_memory_shape, \
                           dtype=numpy_array.dtype, \
                           buffer=existing_shm.buf)
    frames_np[memory_idx_start:memory_idx_end] = numpy_array[:]
    existing_shm.close()
#     return (np.copy(frames_np), correct_memory_shape)
    
def set_memory_use_allow(lock, process_nr, memory_use_allow, value):
    lock.acquire()
    try:
        memory_use_allow[process_nr] = value
    except:
        raise
    finally:
        lock.release()

# Manage extract frames thread
def manage_extract_frames(all_videos_annotations,
                           q_input_extract_frames,
                           q_frames,
                           q_faces,
                           q_log):
    try:
        keys = all_videos_annotations.keys()
        while True:
            if q_input_extract_frames.qsize() < 1 and q_frames.qsize() < 5:
                key = random.choice(list(keys))
                video_annotation = random.choice(all_videos_annotations[key])
                q_input_extract_frames.put(video_annotation)
            else:
                time.sleep(0.1)
    except Exception as e:
        tb = traceback.format_exc()
        q_log.put((e, tb))
    
# Manage mtcnn thread
def manage_fast_mtcnn(q_frames,
                       q_faces,
                       q_log,
                       memory_shape,
                       memory_use_allow,
                       lock,
                       resize=0.25):
    try:
        fast_mtcnn = FastMTCNN(
            resize=resize,
            margin=14,
            factor=0.6,
            keep_all=True,
            device=device,
            thresholds=[0.95,0.95,0.95]
        )
        video_annotation = {}
        q_log.put('start')
        while True:
            if q_faces.qsize() < 5:
                try:
                    video_annotation = q_frames.get(timeout=10)
                    memory_frames = video_annotation['memory_frames']['REAL']
                    process_nr = video_annotation['memory_frames']['process']['nr']
                    frames_np = get_numpy_from_memory(memory_frames, memory_shape)
                    frames_pil = [Image.fromarray(x) for x in frames_np]
                    boxes, probs = fast_mtcnn(frames_pil)
                    video_annotation['faces'] = {
                        'boxes': boxes,
                        'probs': probs
                    }
                    
                    # Add resized crops for REAL as it is faster
                    resized_crops = return_resized_crops(frames_pil, boxes, resize, q_log)
                    if resized_crops:
                        video_annotation['faces']['REAL'] = resized_crops
                        # Crop FAKE with same boxes as REAL
                        memory_frames = video_annotation['memory_frames']['FAKE']
                        frames_np = get_numpy_from_memory(memory_frames, memory_shape)
                        frames_pil = [Image.fromarray(x) for x in frames_np]
                        resized_crops = return_resized_crops(frames_pil, boxes, resize, q_log)
                        video_annotation['faces']['FAKE'] = resized_crops

                        set_memory_use_allow(lock, process_nr, memory_use_allow, 1)
                        q_faces.put(video_annotation)
#                         q_log.put('faces generated by manage_fast_mtcnn')
                    else:
                        set_memory_use_allow(lock, process_nr, memory_use_allow, 1)
#                         q_log.put('More than one face detected for image')
                except queue.Empty:
                    q_log.put('q_frames is empty')
                    pass
                except:
                    raise
            else:
                time.sleep(0.01)
    except Exception as e:
        tb = traceback.format_exc()
        q_log.put((e, tb))
        process_nr = video_annotation['memory_frames']['process']['nr']
        set_memory_use_allow(lock, process_nr, memory_use_allow, 1)
    except:
        tb = traceback.format_exc()
        q_log.put((tb))
        process_nr = video_annotation['memory_frames']['process']['nr']
        set_memory_use_allow(lock, process_nr, memory_use_allow, 1)
        
def manage_batch(q_faces, batches_list, q_log, batch_size=16):
    try:
        ones = np.ones(batch_size)
        zeros = np.zeros(batch_size)
        while True:
            if len(batches_list) < 20:
                video_annotation = q_faces.get()
                crops_faces_real_resized = video_annotation['faces']['REAL']
                crops_faces_fake_resized = video_annotation['faces']['FAKE']
                
                half_batch_size = int(batch_size/2)
                for i in range(int(len(crops_faces_real_resized)/half_batch_size)):
                    s = i*half_batch_size
                    e = (i+1)*half_batch_size
                    X = np.array(crops_faces_fake_resized[s:e] + crops_faces_real_resized[s:e])
                    Y = np.concatenate((np.ones(len(crops_faces_fake_resized[s:e])), \
                                        np.zeros(len(crops_faces_real_resized[s:e]))))
                    
                    if len(X)>0:
                        # Normalize
                        X = (X / 255 - np.array([0.485, 0.456, 0.406])) / np.array([0.229, 0.224, 0.225])

                        X = X.reshape((X.shape[0],3,224,224))
                        Y = np.expand_dims(np.array(Y), axis=1)
                        batches_list.append((X, Y))
            else:
                time.sleep(0.2)
            
    except Exception as e:
        tb = traceback.format_exc()
        q_log.put((e, tb))
    except:
        tb = traceback.format_exc()
        q_log.put((tb))
        
    
        
class MoreThanOneFaceError(Exception):
   """Raised when there is more than 1 face"""
   pass
        
class ExtractFrames:
     
    def __init__(self, 
                 q_input_extract_frames, 
                 q_frames, 
                 q_log, 
                 memory_name,
                 memory_shape,
                 memory_use_allow, 
                 process_nr,
                 lock):
        """
            q_input: Object must have 'path' which directs to from where a video can be opened.

        """
        
#         self.kill_flag = kill_flag
        self.q_input_extract_frames = q_input_extract_frames
        self.q_frames = q_frames
        self.q_log = q_log
        self.memory_name = memory_name
        self.memory_shape = memory_shape
        self.memory_use_allow = memory_use_allow
        self.process_nr = process_nr
        self.lock = lock

    @staticmethod
    def open_video(path):
        v_cap = FileVideoStream(path).start()
        v_len = int(v_cap.stream.get(cv2.CAP_PROP_FRAME_COUNT))

        return (v_cap, v_len)

    @staticmethod
    def get_frames_from_video(v_cap, v_len, double_frames, nr_of_frames, offset):
        frames = []
        for j in range(v_len):
            frame = v_cap.read()
            if frame is not None:
                if j%offset == 0 or (((j+int(offset/2))%offset == 0) and double_frames):
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#                     frame = Image.fromarray(frame)
                    frames.append(frame)
            else:
                break
            if len(frames) >= nr_of_frames:
                if double_frames and len(frames) >= nr_of_frames*2:
                    break
                elif not double_frames:
                    break

        return frames
    
    def get_frames(self, path, nr_of_frames, offset, double_frames):
        video, video_len = self.open_video(path)
        frames = self.get_frames_from_video(video, video_len, nr_of_frames=nr_of_frames, \
                                            offset=offset, double_frames=double_frames)

        return frames
    
    def start(self, resize=0.25, nr_of_frames=20, offset=10, double_frames=True):
        path = str
        try:
            self.q_log.put('start')
            while True:
                skip_video = False
                video_annotation = self.q_input_extract_frames.get()
                path_keys = list(video_annotation.keys())
                video_annotation['memory_frames'] = {}
                idx = 0
                for key in path_keys:
                    path = video_annotation[key]
                    frames_original = np.array(self.get_frames(path, nr_of_frames, offset, double_frames))
                    memory_idx_start = len(frames_original)*idx
                    memory_idx_end = len(frames_original)*(idx+1)
                    while True:
                        try:
                            self.lock.acquire()
                            if self.memory_use_allow[self.process_nr] == 1:
                                put_numpy_array_to_memory(self.memory_name, \
                                                          self.memory_shape, \
                                                          frames_original, \
                                                          memory_idx_start, \
                                                          memory_idx_end)
                                break
                            else:
                                time.sleep(0.2)
                        except Exception as e:
                            tb = traceback.format_exc()
                            self.q_log.put((path, e, tb))
                            skip_video = True
                            break
                        finally:
                            self.lock.release()
                    if skip_video:
                        break
                    video_annotation['memory_frames'][key] = \
                        {'name': self.memory_name,
                        'shape': frames_original.shape,
                        'dtype': frames_original.dtype,
                        'loc_start': memory_idx_start,
                        'loc_end': memory_idx_end
                        }
                    idx = idx + 1
                if not skip_video:
                    set_memory_use_allow(self.lock, self.process_nr, self.memory_use_allow, 0)

                    video_annotation['memory_frames']['process'] = {}
                    video_annotation['memory_frames']['process']['nr'] = self.process_nr
                    self.q_frames.put(video_annotation)

        except Exception as e:
            tb = traceback.format_exc()
            self.q_log.put((path, e, tb))
        except:
            tb = traceback.format_exc()
            self.q_log.put((path, tb))
    
class FastMTCNN(object):
    """Fast MTCNN implementation."""
    
    def __init__(self, resize=1, *args, **kwargs):
        """Constructor for FastMTCNN class.
        
        Arguments:
            stride (int): The detection stride. Faces will be detected every `stride` frames
                and remembered for `stride-1` frames.
        
        Keyword arguments:
            resize (float): Fractional frame scaling. [default: {1}]
            *args: Arguments to pass to the MTCNN constructor. See help(MTCNN).
            **kwargs: Keyword arguments to pass to the MTCNN constructor. See help(MTCNN).
        """
        self.resize = resize
        self.mtcnn = MTCNN(*args, **kwargs)
        
    def __call__(self, frames):
        """Detect faces in frames using strided MTCNN."""
#         frames_original = frames.copy()
        if self.resize != 1:
            frames = [f.resize([int(d * self.resize) for d in f.size]) for f in frames]
                      
        boxes, probs = self.mtcnn.detect(frames)
        
        return (boxes, probs)