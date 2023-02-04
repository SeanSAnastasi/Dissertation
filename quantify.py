import os
import cv2
import numpy as np
import math
import shutil
import random
import matplotlib.pyplot as plt
import sys
import pandas as pd

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
    files = [f for f in os.listdir(directory_pix) if os.path.isfile(os.path.join(directory_pix, f)) and f.endswith(".png")]
    count = 0
    print('HISTOGRAM')
    for file in files:
        # Read the image
        img = cv2.imread(os.path.join(directory_pix, file))
        # Convert the image to HSV color space
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        # Extract the histogram
        hist = cv2.calcHist([hsv], [0, 1, 2], None, [8, 4, 4], [0, 180, 0, 256, 0, 256])
        histograms.append(hist)
        count = count+1
    # print('HISTOGRAM: '+str(count)+'/'+str(len(files)))

    # Perform clustering using KMeans
    print('FITTING')
    histograms = np.array(histograms)
    histograms = histograms.reshape(histograms.shape[0], -1)
    compactness,labels,centers=cv2.kmeans(np.float32(histograms),n,None,(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0),10,cv2.KMEANS_PP_CENTERS)
    # Create the folders
    for i in range(n):
        if not os.path.exists(directory_pix+'/'+str(i+1)):
            os.mkdir(directory_pix+'/'+str(i+1))
        if not os.path.exists(directory_greyscale+'/'+str(i+1)):
            os.mkdir(directory_greyscale+'/'+str(i+1))

    colors = ['red', 'blue', 'green', 'yellow', 'purple', 'orange', 'pink', 'brown', 'grey', 'black']
    df = pd.DataFrame(histograms, columns=["bin_1", "bin_2", "bin_3", "bin_4", "bin_5", "bin_6", "bin_7", "bin_8",
                                            "bin_9", "bin_10", "bin_11", "bin_12", "bin_13", "bin_14", "bin_15", "bin_16",
                                            "bin_17", "bin_18", "bin_19", "bin_20", "bin_21", "bin_22", "bin_23", "bin_24",
                                            "bin_25", "bin_26", "bin_27", "bin_28", "bin_29", "bin_30", "bin_31", "bin_32",
                                            "bin_33", "bin_34", "bin_35", "bin_36", "bin_37", "bin_38", "bin_39", "bin_40",
                                            "bin_41", "bin_42", "bin_43", "bin_44", "bin_45", "bin_46", "bin_47", "bin_48",
                                            "bin_49", "bin_50", "bin_51", "bin_52", "bin_53", "bin_54", "bin_55", "bin_56",
                                            "bin_57", "bin_58", "bin_59", "bin_60", "bin_61", "bin_62", "bin_63", "bin_64",
                                            "bin_65", "bin_66", "bin_67", "bin_68", "bin_69", "bin_70", "bin_71", "bin_72",
                                            "bin_73", "bin_74", "bin_75", "bin_76", "bin_77", "bin_78", "bin_79", "bin_80",
                                            "bin_81", "bin_82", "bin_83", "bin_84", "bin_85", "bin_86", "bin_87", "bin_88",
                                            "bin_89", "bin_90", "bin_91", "bin_92", "bin_93", "bin_94", "bin_95", "bin_96",
                                            "bin_97", "bin_98", "bin_99", "bin_100", "bin_101", "bin_102", "bin_103", "bin_104",
                                            "bin_105", "bin_106", "bin_107", "bin_108", "bin_109", "bin_110", "bin_111", "bin_112",
                                            "bin_113", "bin_114", "bin_115", "bin_116", "bin_117", "bin_118", "bin_119", "bin_120",
                                            "bin_121", "bin_122", "bin_123", "bin_124", "bin_125", "bin_126", "bin_127", "bin_128"])
    df['cluster'] = labels

    from pandas.plotting import parallel_coordinates

    plt.figure()
    parallel_coordinates(df, 'cluster', color=colors)
    plt.title('Parallel Coordinates Plot of Histograms')
    plt.xlabel('Bins')
    plt.ylabel('Value')
    plt.savefig(os.path.join(directory_pix, 'stats', str(n) + '_parallel.png'))

    print('MOVING')
    # Move the images to the corresponding folders
    for i, label in enumerate(labels):
        # print('Iteration: '+str(i)+' Label: '+str(label))
        files2 = [f for f in os.listdir(directory_pix) if os.path.isfile(os.path.join(directory_pix, f))]
        if len(files2) > 0:
            # print('shaw ok')
            try:
                shutil.move(os.path.join(directory_pix, files[i]), directory_pix+'/'+str(label[0]+1))
                shutil.move(os.path.join(directory_greyscale, files[i]), directory_greyscale+'/'+str(label[0]+1))
 
            except:
                print('Label: '+str(label))
                print("error in moving index {}".format(i))
        else:
            break


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
widths = [1600, 1024, 640, 128, 48]
heights = [900, 576, 360, 72, 27]
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

    x = range(1, 20)

    plt.plot(x, similarities_cluster, '-b', label='Cluster')
    plt.plot(x, similarities_random, '-r', label='Random')
    plt.xlabel('Number of clusters')
    plt.ylabel('Similarity Score')
    plt.legend(loc='upper left')
    plt.savefig(film+'/pix'+str(res)+'x'+str(heights[i])+'/stats/chart.png')
