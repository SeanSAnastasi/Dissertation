import sys
import argparse
import os
from PIL import Image
import cv2
from sklearn.cluster import KMeans
import shutil
import numpy as np
from tqdm import tqdm
import gc

def extractImages(pathIn, pathOut):
    count = 0
    if not os.path.exists(pathOut):
        os.makedirs(pathOut)

    vidcap = cv2.VideoCapture(pathIn)
    fps = vidcap.get(cv2.CAP_PROP_FPS)
    duration = vidcap.get(cv2.CAP_PROP_FRAME_COUNT) / fps
    n_frames = int(duration)
    success,image = vidcap.read()
    success = True
    with tqdm(total=n_frames) as pbar:
        while success:
            vidcap.set(cv2.CAP_PROP_POS_MSEC,(count*1000))   
            success,image = vidcap.read()
            if success:
                cv2.imwrite( pathOut + "%d.png" % count, image)    
                pbar.update(1)
            count = count + 1

extractImages(sys.argv[1], sys.argv[2])
gc.collect()
