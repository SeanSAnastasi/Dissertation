import os
import cv2
import numpy as np
import math
import shutil
import random
import matplotlib.pyplot as plt
import sys
import pandas as pd
import tqdm
import plotly.express as px
import plotly.graph_objs as go
from plotly.subplots import make_subplots

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

def rename_png_files(directories):
    for directory in directories:
        for filename in os.listdir(directory):
            if filename.endswith('.png'):
                # Get the original file path and new file path
                original_path = os.path.join(directory, filename)
                new_filename = filename.split('.')[0].zfill(5) + '.png'
                new_path = os.path.join(directory, new_filename)
                # Rename the file
                os.rename(original_path, new_path)

def fit(directory_pix, directory_greyscale, num_dirs, cluster_dir, widths, heights):
    n = num_dirs
    prefix = '1final_'

    # Extract histograms of all the images
    histograms = []
    files = [f for f in tqdm.tqdm(os.listdir(directory_pix+'/'+cluster_dir)) if os.path.isfile(os.path.join(directory_pix+'/'+cluster_dir, f)) and f.endswith(".png")]

    for file in tqdm.tqdm(files):
        # Read the image
        img = cv2.imread(os.path.join(directory_pix+'/'+cluster_dir, file))

        # Convert the image to HSV color space
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        # Extract the histogram
        hist = cv2.calcHist([hsv], [0, 1, 2], None, [8, 4, 4], [0, 180, 0, 256, 0, 256])


        histograms.append(hist)

    # Perform clustering using KMeans
    histograms = np.array(histograms)
    histograms = histograms.reshape(histograms.shape[0], -1)
    compactness,labels,centers=cv2.kmeans(np.float32(histograms),n,None,(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0),10,cv2.KMEANS_PP_CENTERS)

    # Create the folders
    print('MOVING')
    for num, w in enumerate(tqdm.tqdm(widths)):
        for i in range(n):
            if not os.path.exists(directory_pix+f'{w}x{heights[num]}/{prefix}'+str(i+1)):
                os.mkdir(directory_pix+f'{w}x{heights[num]}/{prefix}'+str(i+1))
            if not os.path.exists(directory_greyscale+f'{w}x{heights[num]}/{prefix}'+str(i+1)):
                os.mkdir(directory_greyscale+f'{w}x{heights[num]}/{prefix}'+str(i+1))
            # Get the indices of the images in the current cluster
            cluster_indices = np.where(labels == i)[0]
            # Get the distances of the images in the current cluster to the centroid
            distances = np.linalg.norm(histograms[cluster_indices] - centers[i], axis=1)
            # Sort the images based on their distance to the centroid
            sorted_indices = np.argsort(distances)
            # Rename the images based on their distance to the centroid
            for j, index in enumerate(tqdm.tqdm(cluster_indices[sorted_indices], leave=False)):
                new_filename = f'{j:05}.png'
                # shutil.copyfile(os.path.join(directory_greyscale+f'{w}x{heights[num]}', files[index]), directory_greyscale+f'{w}x{heights[num]}/{prefix}'+str(i+1)+'/'+new_filename)
                shutil.copyfile(os.path.join(directory_pix+f'{w}x{heights[num]}', files[index]), directory_pix+f'{w}x{heights[num]}/{prefix}'+str(i+1)+'/'+new_filename)



def fit_rgb(directory_pix, directory_greyscale, num_dirs, cluster_dir, widths, heights):
    n = num_dirs
    prefix = 'rgb_hist_8_8_8_'

    # Extract histograms of all the images
    print(directory_pix)
    print('FIT INIT')
    histograms = []
    files = [f for f in tqdm.tqdm(os.listdir(directory_pix+'/'+cluster_dir)) if os.path.isfile(os.path.join(directory_pix+'/'+cluster_dir, f)) and f.endswith(".png")]

    print('HISTOGRAM')
    for file in tqdm.tqdm(files):
        # Read the image
        img = cv2.imread(os.path.join(directory_pix+'/'+cluster_dir, file))

        # Convert the image to RGB color space
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # Extract the histogram
        hist = cv2.calcHist([rgb], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
        histograms.append(hist)

    # Perform clustering using KMeans
    print('FITTING')
    histograms = np.array(histograms)
    histograms = histograms.reshape(histograms.shape[0], -1)
    compactness,labels,centers=cv2.kmeans(np.float32(histograms),n,None,(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0),10,cv2.KMEANS_PP_CENTERS)

    # Create the folders
    unique_values = set([item for sublist in labels for item in sublist])
    for num, w in enumerate(widths):
        for i in range(n):
            print(directory_pix+f'{w}x{heights[num]}/{prefix}'+str(i+1))
            if not os.path.exists(directory_pix+f'{w}x{heights[num]}/{prefix}'+str(i+1)):
                os.mkdir(directory_pix+f'{w}x{heights[num]}/{prefix}'+str(i+1))
            if not os.path.exists(directory_greyscale+f'{w}x{heights[num]}/{prefix}'+str(i+1)):
                os.mkdir(directory_greyscale+f'{w}x{heights[num]}/{prefix}'+str(i+1))
        rename_png_files([directory_greyscale+f'{w}x{heights[num]}/'])
        print('MOVING')
        for i, label in enumerate(tqdm.tqdm(labels)):
            try:
                shutil.copyfile(os.path.join(directory_greyscale+f'{w}x{heights[num]}', files[i]), directory_greyscale+f'{w}x{heights[num]}/{prefix}'+str(label[0]+1)+'/'+str(files[i]))
                shutil.copyfile(os.path.join(directory_pix+f'{w}x{heights[num]}', files[i]), directory_pix+f'{w}x{heights[num]}/{prefix}'+str(label[0]+1)+'/'+str(files[i]))
            except:
                pass

def plot_elbow(directory_pix):
    # directory_pix = os.path.join(directory, 'pix/')
    # directory_greyscale = os.path.join(directory, 'greyscale/')

    # Extract histograms of all the images
    print('FIT INIT')

    histograms = []
    files = [f for f in tqdm.tqdm(os.listdir(directory_pix)) if os.path.isfile(os.path.join(directory_pix, f)) and f.endswith(".png")]
    print('HISTOGRAM')
    for file in tqdm.tqdm(files):
        # Read the image
        img = cv2.imread(os.path.join(directory_pix, file))
        # Convert the image to HSV color space
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        # Extract the histogram
        hist = cv2.calcHist([hsv], [0, 1, 2], None, [8, 4, 4], [0, 180, 0, 256, 0, 256])
        # Only include values in the midtone range in the histogram


        histograms.append(hist)

    # Perform clustering using KMeans
    print('FITTING')
    histograms = np.array(histograms)
    histograms = histograms.reshape(histograms.shape[0], -1)
    
    # Compute the within-cluster sum of squares for different number of clusters
    wcss = []
    for i in tqdm.tqdm(range(1, 21)):
        kmeans = cv2.kmeans(np.float32(histograms), i, None, (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0), 10, cv2.KMEANS_PP_CENTERS)
        wcss.append(kmeans[0])
    # Plot the elbow curve
    fig = go.Figure(data=go.Scatter(x=list(range(1, 21)), y=wcss, mode='lines+markers'))
    fig.update_layout(title='Elbow Method for Optimal Number of Clusters',
                      xaxis_title='Number of Clusters', yaxis_title='Within-Cluster Sum of Squares')
    fig.write_html(f'{directory_pix}data/hist_elbow_8_4_4.html')

def move(parent_dir, num_dirs):
    n = num_dirs
    # Define the path to the destination directory
    dest_dir = parent_dir

    # Iterate over all the numeric folders
    for folder in range(1, n+1):
        # Define the path of the current folder
        src_dir = os.path.join(parent_dir, str(folder))
        # Get a list of all the files in the current folder
        files = os.listdir(src_dir)
        # Iterate over all the files in the current folder
        for file in files:
            # Define the path of the current file
            src_file = os.path.join(src_dir, file)
            # Move the file to the destination directory
            shutil.move(src_file, dest_dir)
        # Delete the current folder
        os.rmdir(src_dir)

    print("All images have been moved to the {} folder and numeric folders have been deleted".format(dest_dir))

def fit_random(parent_dir, num_dirs):

    n = num_dirs    

    # Define the path to the source directory
    src_dir = parent_dir

    # Get a list of all the files in the source directory
    files = [f for f in os.listdir(src_dir) if os.path.isfile(os.path.join(src_dir, f)) and f.endswith(".png")]

    # Iterate over all the files in the source directory
    for file in files:
        # Define the path of the current file
        src_file = os.path.join(src_dir, file)
        # Generate a random numeric folder from 1-8
        random_folder = random.randint(1, n)
        # Define the path of the destination directory
        dest_dir = os.path.join(parent_dir, str(random_folder))
        # Create the destination directory if it does not exist
        if not os.path.exists(dest_dir):
            os.makedirs(dest_dir)
        # Move the file to the destination directory
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
k = int(sys.argv[2])
widths = [ 640, 128, 48, 16]
heights = [ 360, 72, 27, 9]
# for i, res in enumerate(widths):
#     similarities_cluster = [] 
#     similarities_random = []
#     if not os.path.exists(film+'/pix/'+str(res)+'x'+str(heights[i])+'/stats/'):
#         os.mkdir(film+'/pix/'+str(res)+'x'+str(heights[i])+'/stats')
#     for num_dirs in range(1, 20):
#         fit(film+'/pix/'+str(res)+'x'+str(heights[i]),film+'/greyscale/'+str(res)+'x'+str(heights[i]), num_dirs)
#         similarities_cluster.append(get_similarity(film+'/pix/'+str(res)+'x'+str(heights[i]), num_dirs))
#         move(film+'/pix/'+str(res)+'x'+str(heights[i]), num_dirs)

#         fit_random(film+'/pix/'+str(res)+'x'+str(heights[i]), num_dirs)
#         similarities_random.append(get_similarity(film+'/pix/'+str(res)+'x'+str(heights[i]), num_dirs))
#         move(film+'/pix/'+str(res)+'x'+str(heights[i]), num_dirs)

#     x = range(1, 20)

#     plt.plot(x, similarities_cluster, '-b', label='Cluster')
#     plt.plot(x, similarities_random, '-r', label='Random')
#     plt.xlabel('Number of clusters')
#     plt.ylabel('Similarity Score')
#     plt.legend(loc='upper left')
#     plt.savefig(film+'/pix/'+str(res)+'x'+str(heights[i])+'/stats/chart.png')
fit(film+'/pix/',film+'/greyscale/', k, '/16x9/',widths, heights)
# plot_elbow(film+'/pix/16x9/')

