import os
import cv2
import numpy as np
import math
import shutil
import random
import matplotlib.pyplot as plt
import sys
import pandas as pd
import plotly.graph_objs as go
import plotly.io as pio
from tqdm import tqdm

def get_similarity(directory, num_dirs):
    n = num_dirs
    # Create a list to store the average similarity scores
    avg_scores = []
    # Iterate over all the numeric folders
    for folder in range(1, n+1):
        # Define the path of the current folder
        path = os.path.join(directory, str(folder))
        files = os.listdir(path)
        if len(files) == 0:
            print("Folder {} is empty".format(folder))
            continue
        # Extract histograms of all the images
        histograms = []
        for file in files:
            if file.endswith(".png"):
                # Read the image
                img = cv2.imread(os.path.join(path, file))
                # Convert the image to HSV color space
                hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
                # Extract the histogram
                hist = cv2.calcHist([hsv], [0, 1, 2], None, [8, 4, 4], [0, 180, 0, 256, 0, 256])
                histograms.append(hist)
        # Compute similarity scores between all the histograms
        similarity_scores = []
        for i in range(len(histograms)):
            for j in range(i+1, len(histograms)):
                # Compute the chi-squared distance between the histograms
                score = cv2.compareHist(histograms[i], histograms[j], cv2.HISTCMP_CHISQR)
                similarity_scores.append(score)
        # Calculate the average similarity score
        avg_score = np.mean(similarity_scores)
        if math.isnan(avg_score):
            print("Folder {} contains less than 2 images, skipping".format(folder))
            continue
        avg_scores.append(avg_score)
        print("Folder {}: Average similarity score: {}".format(folder, avg_score))

    # Calculate the overall average similarity score
    overall_avg_score = np.mean(avg_scores)
    print("Overall average similarity score: {}".format(overall_avg_score))
    return overall_avg_score

def fit(directory_pix, directory_greyscale, num_dirs):
    n = num_dirs
    # directory_pix = os.path.join(directory, 'pix/')
    # directory_greyscale = os.path.join(directory, 'greyscale/')

    # Extract histograms of all the images
    histograms = []
    print('INIT')
    files = [f for f in tqdm(os.listdir(directory_pix)) if os.path.isfile(os.path.join(directory_pix, f)) and f.endswith(".png")]
    count = 0
    print('HISTOGRAM')
    for file in tqdm(files):
        # Read the image
        img = cv2.imread(os.path.join(directory_pix, file))
        # Convert the image to HSV color space
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        # Extract the histogram
        hist = cv2.calcHist([hsv], [0, 1, 2], None, [16, 4, 8], [0, 180, 0, 256, 0, 256])
        histograms.append(hist)
        count = count+1
    # print('HISTOGRAM: '+str(count)+'/'+str(len(files)))

    # Perform clustering using KMeans
    print('FITTING')
    histograms = np.array(histograms)
    histograms = histograms.reshape(histograms.shape[0], -1)
    compactness,labels,centers=cv2.kmeans(np.float32(histograms),n,None,(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0),10,cv2.KMEANS_PP_CENTERS)
    # Create the folders
    for i in tqdm(range(n)):
        if not os.path.exists(directory_pix+'/'+str(i+1)):
            os.mkdir(directory_pix+'/'+str(i+1))
        if not os.path.exists(directory_greyscale+'/'+str(i+1)):
            os.mkdir(directory_greyscale+'/'+str(i+1))

    print('MOVING')
    # Move the images to the corresponding folders
    for i, label in enumerate(tqdm(labels)):

        try:
            shutil.move(os.path.join(directory_pix, files[i]), directory_pix+'/'+str(label[0]+1))
            # shutil.move(os.path.join(directory_greyscale, files[i]), directory_greyscale+'/'+str(label[0]+1))

        except:
            print('Label: '+str(label))
            print("error in moving index {}".format(i))
        


def move(parent_dir, num_dirs):
    # Define the path to the destination directory
    dest_dir = parent_dir

    # Iterate over all the numeric folders
    for folder in  range(1, num_dirs+1):
        # Define the path of the current folder
        src_dir = os.path.join(parent_dir, str(folder))
        # Get a list of all the files in the current folder
        files = os.listdir(src_dir)
        # Move all the files in the current folder to the destination directory
        for file in tqdm(files):
            src_file = os.path.join(src_dir, file)
            shutil.move(src_file, dest_dir)
        # Remove the current folder
        os.rmdir(src_dir)

    print(f"All images have been moved to the {dest_dir} folder and numeric folders have been deleted")

def fit_random(parent_dir, num_dirs):
    # Define the path to the source directory
    src_dir = parent_dir

    # Get a list of all the files in the source directory
    files = [f for f in os.listdir(src_dir) if os.path.isfile(os.path.join(src_dir, f)) and f.endswith(".png")]

    # Generate a list of random numeric folders from 1-8
    dest_dirs = [os.path.join(parent_dir, str(i)) for i in random.sample(range(1, num_dirs+1), len(files))]
    # Create the destination directories if they do not exist
    for dest_dir in dest_dirs:
        if not os.path.exists(dest_dir):
            os.makedirs(dest_dir)

    # Create a list of tuples where each tuple contains the source file path and the destination directory path
    file_dest_pairs = [(os.path.join(src_dir, file), dest_dir) for file, dest_dir in zip(files, dest_dirs)]

    # Move the files to the destination directories in batches
    for i in range(0, len(file_dest_pairs), 100):
        batch = file_dest_pairs[i:i+100]
        src_files, dest_dirs = zip(*batch)
        for src_file, dest_dir in zip(src_files, dest_dirs):
            shutil.move(src_file, dest_dir)
    
    print("All images have been moved to random numeric folders from 1-8")


# move('avatar/pix', 8)
# move('night/greyscale', 8)
# fit('avatar', 8)
# fit('hotel', 8)
# fit('lotr', 8)
# fit('matrix', 8)
# fit('night', 8)




film = sys.argv[1]
widths = [16]
heights = [9]
for i, res in enumerate(widths):
    similarities_cluster = [] 
    similarities_random = []
    if not os.path.exists(film+'/pix/'+str(res)+'x'+str(heights[i])+'/stats/'):
        os.mkdir(film+'/pix/'+str(res)+'x'+str(heights[i])+'/stats')
    for num_dirs in range(1, 20):
        fit(film+'/pix/'+str(res)+'x'+str(heights[i]),film+'/greyscale/'+str(res)+'x'+str(heights[i]), num_dirs)
        similarities_cluster.append(get_similarity(film+'/pix/'+str(res)+'x'+str(heights[i]), num_dirs))
        move(film+'/pix/'+str(res)+'x'+str(heights[i]), num_dirs)

        fit_random(film+'/pix/'+str(res)+'x'+str(heights[i]), num_dirs)
        similarities_random.append(get_similarity(film+'/pix/'+str(res)+'x'+str(heights[i]), num_dirs))
        move(film+'/pix/'+str(res)+'x'+str(heights[i]), num_dirs)

    x = list(range(1, 15))

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=similarities_cluster, name='Cluster', line=dict(color='blue')))
    fig.add_trace(go.Scatter(x=x, y=similarities_random, name='Random', line=dict(color='red')))
    fig.update_layout(xaxis_title='Number of clusters', yaxis_title='Similarity Score', legend=dict(x=0, y=1))
    pio.write_image(fig, film+'/pix'+str(res)+'x'+str(heights[i])+'/stats/chart.png', format='png')
