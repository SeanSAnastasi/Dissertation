import os
import cv2
import plotly.graph_objs as go
import plotly.offline as pyo
import numpy as np
from tqdm import tqdm
import shutil

def postprocess(directory):
    print(directory)
    # Get a list of all image files in the directory
    image_files = [f for f in os.listdir(directory) if f.endswith('.jpg') or f.endswith('.png')]

    # Read the first image in the directory
    img0 = cv2.imread(os.path.join(directory, image_files[0]))
    hsv0 = cv2.cvtColor(img0, cv2.COLOR_BGR2HSV)

    # Calculate the hsv histogram of the first image
    hist0 = cv2.calcHist([hsv0], [0, 1, 2], None, [8, 4, 4], [0, 180, 0, 256, 0, 256])
    cv2.normalize(hist0, hist0, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)

    # Calculate the similarity of each image to the first image
    similarities = []
    for image_file in tqdm(image_files):
        img = cv2.imread(os.path.join(directory, image_file))
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        # Calculate the hsv histogram of the current image
        hist = cv2.calcHist([hsv], [0, 1, 2], None, [8, 4, 4], [0, 180, 0, 256, 0, 256])

        cv2.normalize(hist, hist, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)

        # Calculate the similarity between the current image and the first image
        similarity = cv2.compareHist(hist0, hist, cv2.HISTCMP_CORREL)
        similarity_percent = round(similarity * 100, 2)
        similarities.append(similarity_percent)

    # Delete images with similarity less than 60%
    for i in range(len(similarities)):
        if similarities[i] < 60:
            os.remove(os.path.join(directory, image_files[i]))

def count_images(directory):
    with open(f'{directory}output.txt', 'w') as f:
        for subdir in os.listdir(directory):
            subdir_path = os.path.join(directory, subdir)
            if os.path.isdir(subdir_path):
                images = [file for file in os.listdir(subdir_path) if file.endswith(('jpg', 'jpeg', 'png', 'gif'))]
                f.write(f'{subdir}\t{len(images)}\n')

def summarize_counts(input_file, output_file):
    with open(input_file, 'r') as f:
        lines = f.readlines()
        counts = [int(line.strip().split('\t')[1]) for line in lines]
        means = sum(counts) / len(counts)
        medians = sorted(counts)[len(counts)//2] if len(counts) % 2 == 1 else (sorted(counts)[len(counts)//2 - 1] + sorted(counts)[len(counts)//2]) / 2
        maximums = max(counts)
        minimums = min(counts)
        
    with open(output_file, 'w') as f:
        f.write(f'Mean: {means:.2f}\n')
        f.write(f'Median: {medians:.2f}\n')
        f.write(f'Maximum: {maximums}\n')
        f.write(f'Minimum: {minimums}\n')

def delete_small_dirs(directory, min_images):
    for subdir in os.listdir(directory):
        subdir_path = os.path.join(directory, subdir)
        if os.path.isdir(subdir_path):
            images = [file for file in os.listdir(subdir_path) if file.endswith(('jpg', 'jpeg', 'png', 'gif'))]
            if len(images) < min_images:
                shutil.rmtree(subdir_path)
                print(f"Deleted directory '{subdir_path}' with {len(images)} images")

# Get a list of all subdirectories in the clusters directory
# clusters_dir = 'clusters'
# subdirs = [f.path for f in os.scandir(clusters_dir) if f.is_dir()]

# Iterate over the subdirectories and call the postprocess function on each of them
# for subdir in tqdm(subdirs):
#     postprocess(subdir)

# count_images('clusters/')
# summarize_counts(f'clusters/output.txt', 'clusters/summary.txt')
delete_small_dirs('clusters/', 414)

