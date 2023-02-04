import sys
import argparse
import os
from PIL import Image
import cv2
from sklearn.cluster import KMeans
import shutil
import numpy as np
from tqdm import tqdm

# def extractImages(pathIn, pathOut):
#     count = 0
#     if not os.path.exists(pathOut):
#         os.makedirs(pathOut)

#     vidcap = cv2.VideoCapture(pathIn)
#     fps = vidcap.get(cv2.CAP_PROP_FPS)
#     duration = vidcap.get(cv2.CAP_PROP_FRAME_COUNT) / fps
#     n_frames = int(duration)
#     success,image = vidcap.read()
#     success = True
#     with tqdm(total=n_frames) as pbar:
#         while success:
#             vidcap.set(cv2.CAP_PROP_POS_MSEC,(count*1000))   
#             success,image = vidcap.read()
#             if success:
#                 cv2.imwrite( pathOut + "frame%d.png" % count, image)    
#                 pbar.update(1)
#             count = count + 1

def process_images(film, extension):


    # extractImages('./'+film+extension, film+'/frames/')

    # pixelize

    # directory_og = film+'/frames'
    # directory_pix = film+'/pix'
    # directory_greyscale = film+'/greyscale'
    # directory_resize = film+'/resize'
    # widths = [1600, 1024, 640, 128, 48]
    # heights = [900, 576, 360, 72, 27]
    # count=0

    # os.mkdir(directory_pix)
    # os.mkdir(directory_greyscale)
    # os.mkdir(directory_resize)
    # files = os.listdir(directory_og)



    # for filename in tqdm(files):
    #     img = Image.open(os.path.join(directory_og, filename))


    # # Get the original aspect ratio
    #     width, height = img.size
    #     aspect_ratio = width / height

    #     # Calculate the new width and height to make the aspect ratio 16:9
    #     if aspect_ratio < 16/9:
    #         new_width = int(height * 16 / 9)
    #         img = img.resize((new_width, height), Image.ANTIALIAS)
    #         img.save(os.path.join(directory_resize, filename))

    #     else:
    #         new_height = int(width * 9 / 16)
    #         img = img.resize((width, new_height), Image.ANTIALIAS)
    #         img.save(os.path.join(directory_resize, filename))


    # files = os.listdir(directory_resize) 
    # print('Pixelize')
    # for i, res in enumerate(widths):
    #     os.mkdir(os.path.join(directory_pix, str(res)+'x'+str(heights[i])))
    #     os.mkdir(os.path.join(directory_greyscale, str(res)+'x'+str(heights[i])))

    #     for filename in tqdm(files):
    #         f = os.path.join(directory_og, filename)
    #         # checking if it is a file
    #         if os.path.isfile(f) and os.path.exists(directory_pix):
    #             # Load the image
    #             img = cv2.imread(f)
    #             height, width = img.shape[:2]

    #             # Downsample the image using Lanczos interpolation
    #             img_downsampled = cv2.resize(img, (res, heights[i]), interpolation=cv2.INTER_LANCZOS4)

    #             # Save the downsampled image
    #             cv2.imwrite(directory_pix+'/'+filename, img_downsampled)
    #             # os.remove(f)

    #             count = count+1
    #         elif not os.path.exists(directory_pix):
    #             print('Path doesnt exist')
    #             break
    # count = 0
    # print('GREYSCALE')
    # for i, res in enumerate(widths):
    #     for filename in tqdm(os.listdir(directory_pix)):
    #         f = os.path.join(directory_pix, filename)
    #         # checking if it is a file
    #         if os.path.isfile(f) and os.path.exists(directory_greyscale):
    #             img = Image.open(f).convert('L')
    #             img.save(os.path.join(directory_greyscale, str(res)+'x'+str(heights[i]))+'/'+filename)
    #             count = count+1
    #         elif not os.path.exists(directory_pix):
    #             print('Path doesnt exist')
    #             break

process_images('avatar','.mp4')
process_images('hotel','.mp4')
process_images('lotr','.mkv')
process_images('matrix','.mp4')
process_images('night','.mp4')

# # Define the directory_pix path
# directory_pix = "night/pix"
# directory_grey = "night/greyscale"
# n = 8

# # Extract histograms of all the images
# histograms = []
# files = os.listdir(directory_pix)
# count = 0
# print('HISTOGRAM')
# for file in tqdm(files):
#     if file.endswith(".png"):
#         # Read the image
#         img = cv2.imread(os.path.join(directory_pix, file))
#         # Convert the image to HSV color space
#         hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
#         # Extract the histogram
#         hist = cv2.calcHist([hsv], [0, 1, 2], None, [8, 4, 4], [0, 180, 0, 256, 0, 256])
#         histograms.append(hist)
#         count = count+1
#     # print('HISTOGRAM: '+str(count)+'/'+str(len(files)))

# # Compute similarity scores between all the histograms
# similarity_scores = []
# print('SIMILARITIES')
# for i in tqdm(range(len(histograms))):
#     for j in range(i+1, len(histograms)):
#         # Compute the similarity score using the chi-squared distance
#         score = cv2.compareHist(histograms[i], histograms[j], cv2.HISTCMP_CHISQR)
#         similarity_scores.append((i, j, score))
#     # print('SIMILARITIES: '+str(i)+'/'+str(len(histograms)))

# # Perform clustering using KMeans
# print('FITTING')
# compactness,labels,centers=cv2.kmeans(np.float32(similarity_scores),n,None,(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0),10,cv2.KMEANS_PP_CENTERS)
# print('MOVING')
# # Create the folders
# for i in range(n):
#     os.mkdir(directory_pix+'/'+str(i+1))
#     os.mkdir(directory_grey+'/'+str(i+1))
# # Move the images to the corresponding folders
# for i, label in tqdm(enumerate(labels)):
#     # print('Iteration: '+str(i)+' Label: '+str(label))
#     files2 = [f for f in os.listdir(directory_pix) if os.path.isfile(os.path.join(directory_pix, f))]
#     if len(files2) > 0:
#         # print('shaw ok')
#         try:
#             shutil.move(os.path.join(directory_pix, files[i]), directory_pix+'/'+str(label[0]+1))
#             shutil.move(os.path.join(directory_grey, files[i]), directory_grey+'/'+str(label[0]+1))

#         except:
#             print('Label: '+str(label))
#             print("error in moving index {}".format(i))
#     else:
#         break