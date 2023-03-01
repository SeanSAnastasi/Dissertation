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

film = sys.argv[1]

directory_og = film+'/frames'
directory_pix = film+'/pix'
directory_greyscale = film+'/greyscale'
# widths = [ 640, 128, 48]
# heights = [360, 72, 27]
widths = [ 16]
heights = [9]
count=0

if not os.path.exists(directory_pix):
    os.mkdir(directory_pix)
if not os.path.exists(directory_greyscale):
    os.mkdir(directory_greyscale)

files = os.listdir(directory_og)
print('Pixelize')
for i, res in enumerate(widths):
    os.mkdir(os.path.join(directory_pix, str(res)+'x'+str(heights[i])))
    os.mkdir(os.path.join(directory_greyscale, str(res)+'x'+str(heights[i])))

    for filename in tqdm(files):
        f = os.path.join(directory_og, filename)
        # checking if it is a file
        if os.path.isfile(f) and os.path.exists(directory_pix):
            # Load the image
            img = cv2.imread(f)
            height, width = img.shape[:2]

            # Downsample the image using Lanczos interpolation
            img_downsampled = cv2.resize(img, (res, heights[i]), interpolation=cv2.INTER_LANCZOS4)

            # Save the downsampled image
            cv2.imwrite(directory_pix+'/'+str(res)+'x'+str(heights[i])+'/'+filename, img_downsampled)
            # os.remove(f)

            count = count+1
        elif not os.path.exists(directory_pix):
            print('Path doesnt exist')
            break
count = 0
print('GREYSCALE')
for i, res in enumerate(widths):
    subdir = os.path.join(directory_pix, str(res)+'x'+str(heights[i]))
    if not os.path.exists(subdir):
        print(f"{subdir} does not exist")
        break
    greyscale_subdir = os.path.join(directory_greyscale, str(res)+'x'+str(heights[i]))
    os.makedirs(greyscale_subdir, exist_ok=True)
    for filename in tqdm(os.listdir(subdir)):
        filepath = os.path.join(subdir, filename)
        if os.path.isfile(filepath):
            img = Image.open(filepath).convert('L')
            greyscale_filepath = os.path.join(greyscale_subdir, os.path.splitext(filename)[0] + '.png')
            img.save(greyscale_filepath)
            count = count + 1
gc.collect()