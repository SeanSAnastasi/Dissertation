import os
import cv2
import numpy as np
from tqdm import tqdm
import csv
import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import math
import plotly.express as px
import plotly.graph_objs as go
from plotly.subplots import make_subplots
import shutil

def array_to_csv(array, file_name):
    print(file_name)
    with open(file_name, 'w', newline='') as file:
        writer = csv.writer(file)
        for row in array:
            writer.writerow(row.flatten())

def cluster_csv_file_h_fast(path, n_clusters, row, col):
    hs = csv_to_array(path+'/h_full.csv', row, col)
    hs = hs.reshape(-1, hs.shape[1] * hs.shape[2])
    print(hs.shape)

    centroids = []
    for i in tqdm(range(hs.shape[0])):
        kmeans = KMeans(n_clusters=n_clusters,n_init=10)
        kmeans.fit(hs[i].reshape(-1, 1))
        centroids.append(np.sort(kmeans.cluster_centers_.flatten()))
    
    centroids = np.array(centroids)
    title = []
    for i in range(n_clusters):
        title.append(f'HUE_{row}_{str(n_clusters)}_{str(i+1)}')
    print(centroids.shape)
    centroids = np.concatenate((np.array([title]), centroids))
    array_to_csv(centroids, path+'/h_'+str(n_clusters)+'.csv')

def cluster_csv_file_s_fast(path, n_clusters, row, col):
    ss = csv_to_array(path+'/s_full.csv', row, col)
    ss = ss.reshape(-1, ss.shape[1] * ss.shape[2])
    print(ss.shape)

    centroids = []
    for i in tqdm(range(ss.shape[0])):
        kmeans = KMeans(n_clusters=n_clusters,n_init=10)
        kmeans.fit(ss[i].reshape(-1, 1))
        centroids.append(np.sort(kmeans.cluster_centers_.flatten()))
    
    centroids = np.array(centroids)
    title = []
    for i in range(n_clusters):
        title.append(f'SATURATION_{row}_{str(n_clusters)}_{str(i+1)}')
    print(centroids.shape)
    centroids = np.concatenate((np.array([title]), centroids))

    print(centroids.shape)
    array_to_csv(centroids, path+'/s_'+str(n_clusters)+'.csv')

    
def cluster_csv_file_v_fast(path, n_clusters, row, col):
    vs = csv_to_array(path+'/v_full.csv', row, col)
    vs = vs.reshape(-1, vs.shape[1] * vs.shape[2])
    print(vs.shape)
    
    centroids = []
    for i in tqdm(range(vs.shape[0])):
        kmeans = KMeans(n_clusters=n_clusters,n_init=10)
        kmeans.fit(vs[i].reshape(-1, 1))
        centroids.append(np.sort(kmeans.cluster_centers_.flatten()))
    
    centroids = np.array(centroids)
    title = []
    for i in range(n_clusters):
        title.append(f'VALUE_{row}_{str(n_clusters)}_{str(i+1)}')
    print(centroids.shape)
    centroids = np.concatenate((np.array([title]), centroids))
    print(centroids.shape)
    array_to_csv(centroids, path+'/v_'+str(n_clusters)+'.csv')

def hsv_to_rgb(h, s, v):
        if s == 0.0: return (v, v, v)
        i = int(h*6.) # XXX assume int() truncates!
        f = (h*6.)-i; p,q,t = v*(1.-s), v*(1.-s*f), v*(1.-s*(1.-f)); i%=6
        if i == 0: return (v, t, p)
        if i == 1: return (q, v, p)
        if i == 2: return (p, v, t)
        if i == 3: return (p, q, v)
        if i == 4: return (t, p, v)
        if i == 5: return (v, p, q)
    
def plot_csv(file_path):
    print(file_path)
    df = pd.read_csv(file_path)
    if df.shape[1] > 1:
        df = df.apply(lambda x: np.sort(x), axis=1, raw=True)

    columns = df.columns
    fig, ax = plt.subplots()
    fig.set_size_inches(300, 5)
    ax.set_aspect('auto')
    for column in columns:
        hue_value = int(column.replace("hue_", ""))
        color = hsv_to_rgb((hue_value / 360, 1.0, 1.0))
        ax.plot(df.index, df[column], label=column, color=color)
    ax.set_xlabel("Index")
    ax.set_ylabel("Value")
    ax.legend()
    plt.tight_layout()
    save_path = file_path.replace(".csv", ".png")
    plt.savefig(save_path)

def plot_csv_interactive(input_files, output_file):
    # file_path = './'+film+'/pix/'+str(width)+'x'+str(height)+'/data/'
    # files = [f'h_{cluster}{postfix}.csv', f's_{cluster}{postfix}.csv', f'v_{cluster}{postfix}.csv']
    files = input_files

    # create an empty dataframe to store the final result
    df = pd.DataFrame()

    # loop through the list of files and concatenate them one by one
    for file in files:
        df_temp = pd.read_csv(file)
        df = pd.concat([df, df_temp], axis=1)
    columns = df.columns
    traces = []
    line_colors = ['red', 'blue', 'green', 'maroon', 'darkslateblue', 'lavender', 'silver', 'saddlebrown', 'darkslategrey', 'violet', 'seashell', 'lightcoral', 'cadetblue', 'dimgrey', 'hotpink', 'lightcyan', 'cyan', 'plum', 'deepskyblue', 'rosybrown', 'navy', 'gainsboro', 'magenta', 'lightsteelblue', 'indianred', 'mistyrose', 'darkgreen', 'slategray', 'thistle', 'purple', 'lavenderblush', 'lightgreen', 'palevioletred', 'azure', 'darkgoldenrod', 'linen', 'chartreuse', 'mediumseagreen', 'lemonchiffon', 'coral', 'indigo', 'lightslategrey', 'gold', 'lime', 'mediumorchid', 'wheat', 'blueviolet', 'fuchsia', 'lawngreen', 'orange', 'mediumblue', 'moccasin', 'powderblue', 'darkolivegreen', 'sienna', 'springgreen', 'cornsilk', 'peachpuff', 'khaki', 'orchid', 'dodgerblue', 'palegreen', 'yellowgreen', 'goldenrod', 'tomato', 'deeppink', 'aquamarine', 'teal', 'lightslategray', 'limegreen', 'darkcyan', 'mediumpurple', 'green', 'beige', 'darkturquoise', 'firebrick', 'aquamarine', 'whitesmoke', 'blue', 'cornflowerblue', 'lightyellow', 'yellow', 'mediumslateblue', 'greenyellow', 'black', 'mistyrose', 'papayawhip', 'burlywood', 'darkmagenta', 'forestgreen', 'lightpink', 'lightgoldenrodyellow', 'peru', 'brown', 'pink', 'darkorange', 'darkslateblue', 'goldenrod', 'mediumspringgreen', 'snow', 'lightsalmon', 'crimson', 'oldlace', 'lightblue', 'purple', 'olive', 'aquamarine', 'blanchedalmond', 'mediumaquamarine', 'antiquewhite', 'magenta', 'navajowhite', 'tan', 'midnightblue', 'sandybrown', 'ghostwhite', 'honeydew', 'darkgrey', 'darkred', 'deeppink', 'palegoldenrod', 'darkkhaki', 'yellow', 'lightgreen', 'darkorchid', 'cyan', 'skyblue', 'mediumturquoise', 'darkseagreen', 'darkgrey', 'mediumslateblue', 'aquamarine', 'indigo', 'blue', 'ivory', 'burlywood', 'mediumvioletred', 'whitesmoke', 'lightblue', 'pink', 'mediumspringgreen', 'moccasin', 'goldenrod', 'magenta', 'lavenderblush', 'paleturquoise', 'cornflowerblue', 'darkgreen', 'mediumseagreen', 'palevioletred', 'palegreen', 'skyblue', 'lightskyblue', 'olivedrab', 'dimgrey', 'mediumslateblue', 'darkolivegreen', 'orchid', 'mediumblue', 'grey', 'lime', 'honeydew']
    for i,col in enumerate(columns):
        try:
            y = df[col].tolist()
        
            colors = []
            for j,y_i in enumerate(y):
                hue = int(df.loc[j, col.replace("Saturation", "Hue").replace("Value", "Hue")])
                saturation = int(df.loc[j, col.replace("Hue", "Saturation").replace("Value", "Saturation")])
                value = int(df.loc[j, col.replace("Saturation", "Value").replace("Hue", "Value")])

                colors.append(f'hsv({hue},{int(saturation)},{int(value)})')

            trace = go.Scatter(x=df.index, y=y, mode='markers+lines', name=col, marker={'color': colors},line={'color': line_colors[i%len(line_colors)]})
            traces.append(trace)
        except:
            print(df[col])
            return

    layout = go.Layout(title='Interactive Line Plot')
    fig = go.Figure(data=traces, layout=layout)
    fig.write_html(output_file)

def plot_scatter_interactive(input_file, output_file, color_files=None):
    # Load data from CSV file
    data = pd.read_csv(input_file)
    hue = pd.read_csv(color_files[0])
    saturation = pd.read_csv(color_files[1])
    value = pd.read_csv(color_files[2])
    colors = []

    colors = []
    for j,row in hue.iterrows():
        h = int(row.iloc[0])
        s = saturation.iloc[j,0]
        v = value.iloc[j,0]

        colors.append(f'hsv({h},{int(s)},{int(v)})')

    # Determine number of dimensions based on number of columns
    num_cols = data.shape[1]
    if num_cols == 1:
        # 1D scatter plot
        fig = go.Figure(data=go.Scatter(x=data.iloc[:, 0], mode='markers',marker={'color': colors}, name=data.columns[0]))
    elif num_cols == 2:
        # 2D scatter plot
        fig = go.Figure(data=go.Scatter(x=data.iloc[:, 0], y=data.iloc[:, 1], mode='markers',marker={'color': colors}, name=data.columns[1]))
    elif num_cols == 3:
        # 3D scatter plot
        fig = go.Figure(data=[go.Scatter3d(x=data.iloc[:, 0], y=data.iloc[:, 1], z=data.iloc[:, 2], mode='markers',marker={'color': colors}, name=data.columns[2])])
        fig.update_layout(scene=dict(xaxis_title=data.columns[0], yaxis_title=data.columns[1], zaxis_title=data.columns[2]))
    elif num_cols == 4:
        # 4D scatter plot
        fig = make_subplots(rows=1, cols=1, specs=[[{'type': 'scatter'}]])
        fig.add_trace(go.Scatter(x=data.iloc[:, 0], y=data.iloc[:, 1], mode='markers',
                                 marker=dict(color=colors, colorscale='Viridis'), name=data.columns[1]), row=1, col=1)
        fig.update_layout(xaxis_title=data.columns[0], yaxis_title=data.columns[1],
                          coloraxis=dict(colorscale='Viridis', colorbar=dict(title=data.columns[3])))
    else:
        raise ValueError("Input file must have 1 to 4 columns")

    # Add legend
    fig.update_layout(showlegend=True)

    # Save scatter plot as an HTML file
    fig.write_html(output_file)

def create_csvs(path):
    # Get all files in the directory
    print('GETTING FILES')
    files = [f for f in tqdm(os.listdir(path)) if os.path.isfile(os.path.join(path, f))]
    # Sort the files by name
    files.sort()

    print('POPULATING ARRAYS')
    
    hs_list = []
    ss_list = []
    vs_list = []
    
    for file in tqdm(files):
        # Only process PNG images
        if file.endswith(".png"):
            # Load the image
            img = cv2.imread(os.path.join(path, file))
            # Convert the image to the HSV color space
            hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            # Split the image into separate channels (hue, saturation, value)
            h, s, v = cv2.split(hsv)
            hs_list.append(h)
            ss_list.append(s)
            vs_list.append(v)

    hs = np.stack(hs_list, axis=0)
    ss = np.stack(ss_list, axis=0)
    vs = np.stack(vs_list, axis=0)

    if not os.path.exists(path+'/data'):
        os.mkdir(path+'/data')
    array_to_csv(hs, path+'/data/h_full.csv')
    array_to_csv(ss, path+'/data/s_full.csv')
    array_to_csv(vs, path+'/data/v_full.csv')

def csv_to_array(file_name, rows, cols):
    with open(file_name, 'r') as file:
        reader = csv.reader(file)
        data = np.fromiter((int(x) for row in tqdm(reader) for x in row), dtype=int)
    return data.reshape(-1, cols, rows)

def csv_to_array_half(file_name, rows, cols, first_half=True):
    with open(file_name, 'r') as file:
        reader = csv.reader(file)
        data = []
        row_count = 0
        for row in reader:
            if first_half:
                if row_count >= rows // 2:
                    break
            else:
                if row_count < rows // 2:
                    row_count += 1
                    continue
            data += [int(x) for x in row]
            row_count += 1
    return np.array(data).reshape(-1, cols, rows)

# Trades off speed for memory usage
def cluster_csv_file_h_slow(path, n_clusters, row, col):
    print('CLUSTERING')
    centroids = []
    with open(path+'/h_full.csv', 'r') as file:
        reader = csv.reader(file)
        for i, row in enumerate(tqdm(reader)):
            data = np.array([int(x) for x in row])
            data = data.reshape(-1, col)
            kmeans = KMeans(n_clusters=n_clusters, n_init=10)
            kmeans.fit(data)
            centroids.append(kmeans.cluster_centers_.flatten())
    centroids = np.array(centroids)
    print(centroids.shape)
    array_to_csv(centroids, path+'/h_'+str(n_clusters)+'.csv')
# Trades off memory usage for speed


    


def rename_files(path):
    # Get all files in the directory
    print('Getting files')

    files = [f for f in tqdm(os.listdir(path)) if os.path.isfile(os.path.join(path, f))]
    for file in tqdm(files):
        # Get the file name without the extension
        file_name, file_ext = os.path.splitext(file)
        # Pad the file name with zeros to make it 5 characters long
        padded_file_name = file_name.zfill(5)
        # Construct the new file name
        new_file_name = padded_file_name + file_ext
        # Rename the file
        os.rename(os.path.join(path, file), os.path.join(path, new_file_name))

def smooth_csv_file(file_path, window):
    df = pd.read_csv(file_path)
    smooth_df = df.rolling(window).mean()
    smooth_df = smooth_df.fillna(0)
    new_cols = [col + '_smooth_' + str(window) for col in smooth_df.columns]
    smooth_df.columns = new_cols
    new_file_path = file_path.split(".csv")[0] + f"_smooth_{str(window)}.csv"
    smooth_df.to_csv(new_file_path, index=False)

def calculate_avg_distance_to_centroid(file_paths, k):
    # Read in the CSV files as pandas DataFrames
    data_frames = []
    for path in file_paths:
        data_frames.append(pd.read_csv(path))

    # Concatenate the DataFrames along columns
    data = pd.concat(data_frames, axis=1)

    # Perform k-means clustering on the data
    kmeans = KMeans(n_clusters=k, init='k-means++', max_iter=300, n_init=10, random_state=0).fit(data)

    return kmeans.inertia_

def cluster(file_paths,image_path, film, k):
    print('CLUSTERING')
    # Read in the CSV files as pandas DataFrames
    data_frames = []
    greyscale_path = f'{film}/greyscale'
    for path in file_paths:
        data_frames.append(pd.read_csv(path))

    # Concatenate the DataFrames along columns
    data = pd.concat(data_frames, axis=1)

    # Perform k-means clustering on the data
    kmeans = KMeans(n_clusters=k, n_init=10).fit(data)

    for i in tqdm(range(len(data))):
        row_number = str(i).zfill(5) + '.png'
        row_number_g = str(i)+'.png'
        cluster_label = str(kmeans.labels_[i])

        widths=[16, 48, 128, 640]
        heights=[9, 27, 72, 360]
        for j, r in enumerate(widths):
            # Create directory if it doesn't exist
            dir_path = os.path.join(image_path,f'{r}x{heights[j]}', cluster_label)
            dir_path_g = os.path.join(greyscale_path,f'{r}x{heights[j]}', cluster_label)
            if not os.path.exists(dir_path):
                os.makedirs(dir_path)
                os.makedirs(dir_path_g)

            # Copy image to appropriate directory
            src_path = os.path.join(image_path,f'{r}x{heights[j]}', row_number)
            src_path_g = os.path.join(greyscale_path,f'{r}x{heights[j]}', row_number_g)

            dst_path = os.path.join(dir_path, row_number)
            dst_path_g = os.path.join(dir_path_g, row_number_g)

            shutil.copyfile(src_path, dst_path)
            shutil.copyfile(src_path_g, dst_path_g)


    print('Images copied to cluster directories.')


# ,
films_fast = [ 'avatar', 'blade', 'gbu', 'godfather', 'martian', 'matrix',  'night', 'scan', 'lotr']
ks = [4, 7,4, 4, 6, 4, 4, 5, 4]
# films_slow = ['lotr']
# films_fast = ['matrix']
# films_fast = ['avatar']

widths=[16, 48, 128]
heights=[9, 27, 72]
# HAVENT DONE ANY 640 BESIDES MATRIX
for n, film in enumerate(films_fast):
    for j, r in enumerate(widths):
        for i in range(1, 5):
            print('FILM: '+film+ ' RES: '+str(r)+' CLUSTER: '+str(i))
            # cluster_csv_file_h_fast('./'+film+'/pix/'+str(r)+'x'+str(heights[j])+'/data', i, r, heights[j])
            # cluster_csv_file_s_fast('./'+film+'/pix/'+str(r)+'x'+str(heights[j])+'/data', i, r, heights[j])
            # cluster_csv_file_v_fast('./'+film+'/pix/'+str(r)+'x'+str(heights[j])+'/data', i, r, heights[j])
            # smooth_csv_file('./'+film+'/pix/'+str(r)+'x'+str(heights[j])+'/data/h_'+str(i)+'.csv', 4)
            # smooth_csv_file('./'+film+'/pix/'+str(r)+'x'+str(heights[j])+'/data/s_'+str(i)+'.csv', 4)
            # smooth_csv_file('./'+film+'/pix/'+str(r)+'x'+str(heights[j])+'/data/v_'+str(i)+'.csv', 4)
            # smooth_csv_file('./'+film+'/pix/'+str(r)+'x'+str(heights[j])+'/data/h_'+str(i)+'.csv', 10)
            # smooth_csv_file('./'+film+'/pix/'+str(r)+'x'+str(heights[j])+'/data/s_'+str(i)+'.csv', 10)
            # smooth_csv_file('./'+film+'/pix/'+str(r)+'x'+str(heights[j])+'/data/v_'+str(i)+'.csv', 10)

            # path_1 = f'./{film}/pix/{str(r)}x{str(heights[j])}/data/'
            # plot_csv_interactive(input_files=[f'{path_1}h_{str(i)}.csv', f'{path_1}s_{str(i)}.csv', f'{path_1}v_{str(i)}.csv'], output_file=f'{path_1}cluster_{str(i)}.html')
            # plot_csv_interactive(input_files=[f'{path_1}h_{str(i)}_smooth_4.csv', f'{path_1}s_{str(i)}_smooth_4.csv', f'{path_1}v_{str(i)}_smooth_4.csv'], output_file=f'{path_1}cluster_{str(i)}_smooth_4.html')
            # plot_csv_interactive(input_files=[f'{path_1}h_{str(i)}_smooth_10.csv', f'{path_1}s_{str(i)}_smooth_10.csv', f'{path_1}v_{str(i)}_smooth_10.csv'], output_file=f'{path_1}cluster_{str(i)}_smooth_10.html')
            # plot_csv_interactive(input_files=[f'{path_1}h_{str(i)}.csv',f'{path_1}h_{str(i)}_smooth_4.csv', f'{path_1}h_{str(i)}_smooth_10.csv'], output_file=f'{path_1}cluster_{str(i)}_compare_smooth_h.html')
            # plot_csv_interactive(input_files=[f'{path_1}s_{str(i)}.csv',f'{path_1}s_{str(i)}_smooth_4.csv', f'{path_1}s_{str(i)}_smooth_10.csv'], output_file=f'{path_1}cluster_{str(i)}_compare_smooth_s.html')
            # plot_csv_interactive(input_files=[f'{path_1}v_{str(i)}.csv',f'{path_1}v_{str(i)}_smooth_4.csv', f'{path_1}v_{str(i)}_smooth_10.csv'], output_file=f'{path_1}cluster_{str(i)}_compare_smooth_v.html')
            
            # plot_scatter_interactive(input_file=f'{path_1}h_{str(i)}.csv', output_file=f'{path_1}scatter_h_{str(i)}.html' , color_files=[f'{path_1}h_1.csv', f'{path_1}s_1.csv', f'{path_1}v_1.csv'])
            # plot_scatter_interactive(input_file=f'{path_1}h_{str(i)}.csv', output_file=f'{path_1}scatter_h_{str(i)}_smooth_4.html', color_files=[f'{path_1}h_1_smooth_4.csv', f'{path_1}s_1_smooth_4.csv', f'{path_1}v_1_smooth_4.csv'] )
            # plot_scatter_interactive(input_file=f'{path_1}h_{str(i)}.csv', output_file=f'{path_1}scatter_h_{str(i)}_smooth_10.html', color_files=[f'{path_1}h_1_smooth_10.csv', f'{path_1}s_1_smooth_10.csv', f'{path_1}v_1_smooth_10.csv'] )

            # plot_scatter_interactive(input_file=f'{path_1}s_{str(i)}.csv', output_file=f'{path_1}scatter_s_{str(i)}.html', color_files=[f'{path_1}h_1.csv', f'{path_1}s_1.csv', f'{path_1}v_1.csv'] )
            # plot_scatter_interactive(input_file=f'{path_1}s_{str(i)}.csv', output_file=f'{path_1}scatter_s_{str(i)}_smooth_4.html', color_files=[f'{path_1}h_1_smooth_4.csv', f'{path_1}s_1_smooth_4.csv', f'{path_1}v_1_smooth_4.csv'] )
            # plot_scatter_interactive(input_file=f'{path_1}s_{str(i)}.csv', output_file=f'{path_1}scatter_s_{str(i)}_smooth_10.html', color_files=[f'{path_1}h_1_smooth_10.csv', f'{path_1}s_1_smooth_10.csv', f'{path_1}v_1_smooth_10.csv'] )

            # plot_scatter_interactive(input_file=f'{path_1}v_{str(i)}.csv', output_file=f'{path_1}scatter_v_{str(i)}.html', color_files=[f'{path_1}h_1.csv', f'{path_1}s_1.csv', f'{path_1}v_1.csv'] )
            # plot_scatter_interactive(input_file=f'{path_1}v_{str(i)}.csv', output_file=f'{path_1}scatter_v_{str(i)}_smooth_4.html', color_files=[f'{path_1}h_1_smooth_4.csv', f'{path_1}s_1_smooth_4.csv', f'{path_1}v_1_smooth_4.csv'] )
            # plot_scatter_interactive(input_file=f'{path_1}v_{str(i)}.csv', output_file=f'{path_1}scatter_v_{str(i)}_smooth_10.html', color_files=[f'{path_1}h_1_smooth_10.csv', f'{path_1}s_1_smooth_10.csv', f'{path_1}v_1_smooth_10.csv'] )

            # print('GETTING ELBOW')
            # h_avg = []
            # h_avg_4 = []
            # h_avg_10 = []
            # s_avg = []
            # s_avg_4 = []
            # s_avg_10 = []
            # v_avg = []
            # v_avg_4 = []
            # v_avg_10 = []

            # hs_avg = []
            # hs_avg_4 = []
            # hs_avg_10 = []
            # hsv_avg = []
            # hsv_avg_4 = []
            # hsv_avg_10 = []
            # for k in range(1, 11):
            #     print(f'CLUSTER: {k}')
            #     h_avg.append(calculate_avg_distance_to_centroid([f'{path_1}h_{str(i)}.csv'], k))
            #     h_avg_4.append(calculate_avg_distance_to_centroid([f'{path_1}h_{str(i)}_smooth_4.csv'], k))
            #     h_avg_10.append(calculate_avg_distance_to_centroid([f'{path_1}h_{str(i)}_smooth_10.csv'], k))
            #     s_avg.append(calculate_avg_distance_to_centroid([f'{path_1}s_{str(i)}.csv'], k))
            #     s_avg_4.append(calculate_avg_distance_to_centroid([f'{path_1}s_{str(i)}_smooth_4.csv'], k))
            #     s_avg_10.append(calculate_avg_distance_to_centroid([f'{path_1}s_{str(i)}_smooth_10.csv'], k))
            #     v_avg.append(calculate_avg_distance_to_centroid([f'{path_1}v_{str(i)}.csv'], k))
            #     v_avg_4.append(calculate_avg_distance_to_centroid([f'{path_1}v_{str(i)}_smooth_4.csv'], k))
            #     v_avg_10.append(calculate_avg_distance_to_centroid([f'{path_1}v_{str(i)}_smooth_10.csv'], k))

            #     hs_avg.append(calculate_avg_distance_to_centroid([f'{path_1}h_{str(i)}.csv', f'{path_1}s_{str(i)}.csv'], k))
            #     hs_avg_4.append(calculate_avg_distance_to_centroid([f'{path_1}h_{str(i)}_smooth_4.csv', f'{path_1}s_{str(i)}_smooth_4.csv'], k))
            #     hs_avg_10.append(calculate_avg_distance_to_centroid([f'{path_1}h_{str(i)}_smooth_10.csv', f'{path_1}h_{str(i)}_smooth_10.csv'], k))
            #     hsv_avg.append(calculate_avg_distance_to_centroid([f'{path_1}h_{str(i)}.csv', f'{path_1}s_{str(i)}.csv', f'{path_1}v_{str(i)}.csv'], k))
            #     hsv_avg_4.append(calculate_avg_distance_to_centroid([f'{path_1}h_{str(i)}_smooth_4.csv', f'{path_1}s_{str(i)}_smooth_4.csv', f'{path_1}v_{str(i)}_smooth_4.csv'], k))
            #     hsv_avg_10.append(calculate_avg_distance_to_centroid([f'{path_1}h_{str(i)}_smooth_10.csv', f'{path_1}h_{str(i)}_smooth_10.csv', f'{path_1}v_{str(i)}_smooth_10.csv'], k))


            # fig = go.Figure(data=go.Scatter(x=list(range(len(h_avg))), y=h_avg, mode='lines'))
            # fig.update_layout(title='Plot of Array Values', xaxis_title='Index', yaxis_title='Value')
            # fig.write_html(f'{path_1}elbow_h_{str(i)}.html')

            # fig = go.Figure(data=go.Scatter(x=list(range(len(h_avg_4))), y=h_avg_4, mode='lines'))
            # fig.update_layout(title='Plot of Array Values', xaxis_title='Index', yaxis_title='Value')
            # fig.write_html(f'{path_1}elbow_h_{str(i)}_4.html')

            # fig = go.Figure(data=go.Scatter(x=list(range(len(h_avg_10))), y=h_avg_10, mode='lines'))
            # fig.update_layout(title='Plot of Array Values', xaxis_title='Index', yaxis_title='Value')
            # fig.write_html(f'{path_1}elbow_h_{str(i)}_10.html')

            # fig = go.Figure(data=go.Scatter(x=list(range(len(s_avg))), y=s_avg, mode='lines'))
            # fig.update_layout(title='Plot of Array Values', xaxis_title='Index', yaxis_title='Value')
            # fig.write_html(f'{path_1}elbow_s_{str(i)}.html')

            # fig = go.Figure(data=go.Scatter(x=list(range(len(s_avg_4))), y=s_avg_4, mode='lines'))
            # fig.update_layout(title='Plot of Array Values', xaxis_title='Index', yaxis_title='Value')
            # fig.write_html(f'{path_1}elbow_s_{str(i)}_4.html')

            # fig = go.Figure(data=go.Scatter(x=list(range(len(s_avg_10))), y=s_avg_10, mode='lines'))
            # fig.update_layout(title='Plot of Array Values', xaxis_title='Index', yaxis_title='Value')
            # fig.write_html(f'{path_1}elbow_s_{str(i)}_10.html')

            # fig = go.Figure(data=go.Scatter(x=list(range(len(v_avg))), y=v_avg, mode='lines'))
            # fig.update_layout(title='Plot of Array Values', xaxis_title='Index', yaxis_title='Value')
            # fig.write_html(f'{path_1}elbow_v_{str(i)}.html')

            # fig = go.Figure(data=go.Scatter(x=list(range(len(v_avg_4))), y=v_avg_4, mode='lines'))
            # fig.update_layout(title='Plot of Array Values', xaxis_title='Index', yaxis_title='Value')
            # fig.write_html(f'{path_1}elbow_v_{str(i)}_4.html')

            # fig = go.Figure(data=go.Scatter(x=list(range(len(v_avg_10))), y=v_avg_10, mode='lines'))
            # fig.update_layout(title='Plot of Array Values', xaxis_title='Index', yaxis_title='Value')
            # fig.write_html(f'{path_1}elbow_v_{str(i)}_10.html')

            # fig = go.Figure(data=go.Scatter(x=list(range(len(hs_avg))), y=hs_avg, mode='lines'))
            # fig.update_layout(title='Plot of Array Values', xaxis_title='Index', yaxis_title='Value')
            # fig.write_html(f'{path_1}elbow_hs_{str(i)}.html')

            # fig = go.Figure(data=go.Scatter(x=list(range(len(hs_avg_4))), y=hs_avg_4, mode='lines'))
            # fig.update_layout(title='Plot of Array Values', xaxis_title='Index', yaxis_title='Value')
            # fig.write_html(f'{path_1}elbow_hs_{str(i)}_4.html')

            # fig = go.Figure(data=go.Scatter(x=list(range(len(hs_avg_10))), y=hs_avg_10, mode='lines'))
            # fig.update_layout(title='Plot of Array Values', xaxis_title='Index', yaxis_title='Value')
            # fig.write_html(f'{path_1}elbow_hs_{str(i)}_10.html')

            # fig = go.Figure(data=go.Scatter(x=list(range(len(hsv_avg))), y=hsv_avg, mode='lines'))
            # fig.update_layout(title='Plot of Array Values', xaxis_title='Index', yaxis_title='Value')
            # fig.write_html(f'{path_1}elbow_hsv_{str(i)}.html')

            # fig = go.Figure(data=go.Scatter(x=list(range(len(hsv_avg_4))), y=hsv_avg_4, mode='lines'))
            # fig.update_layout(title='Plot of Array Values', xaxis_title='Index', yaxis_title='Value')
            # fig.write_html(f'{path_1}elbow_hsv_{str(i)}_4.html')

            # fig = go.Figure(data=go.Scatter(x=list(range(len(hsv_avg_10))), y=hsv_avg_10, mode='lines'))
            # fig.update_layout(title='Plot of Array Values', xaxis_title='Index', yaxis_title='Value')
            # fig.write_html(f'{path_1}elbow_hsv_{str(i)}_10.html')

    path_1 = f'./{film}/pix/16x9/data/'
    path_2 = f'./{film}/pix/48x27/data/'
    path_3 = f'./{film}/pix/128x72/data/'
    # path_4 = f'./{film}/pix/640x360/data/'
    output = f'./{film}/pix/'
    # plot_csv_interactive(input_files=[f'{path_1}h_1.csv', f'{path_1}s_1.csv', f'{path_1}v_1.csv',f'{path_2}h_1.csv', f'{path_2}s_1.csv', f'{path_2}v_1.csv',f'{path_3}h_1.csv', f'{path_3}s_1.csv', f'{path_3}v_1.csv',f'{path_4}h_1.csv', f'{path_4}s_1.csv', f'{path_4}v_1.csv'], output_file=f'{output}compare_1.html')
    # plot_csv_interactive(input_files=[f'{path_1}h_2.csv', f'{path_1}s_2.csv', f'{path_1}v_2.csv',f'{path_2}h_2.csv', f'{path_2}s_2.csv', f'{path_2}v_2.csv',f'{path_3}h_2.csv', f'{path_3}s_2.csv', f'{path_3}v_2.csv',f'{path_4}h_2.csv', f'{path_4}s_2.csv', f'{path_4}v_2.csv'], output_file=f'{output}compare_2.html')
    # plot_csv_interactive(input_files=[f'{path_1}h_3.csv', f'{path_1}s_3.csv', f'{path_1}v_3.csv',f'{path_2}h_3.csv', f'{path_2}s_3.csv', f'{path_2}v_3.csv',f'{path_3}h_3.csv', f'{path_3}s_3.csv', f'{path_3}v_3.csv',f'{path_4}h_3.csv', f'{path_4}s_3.csv', f'{path_4}v_3.csv'], output_file=f'{output}compare_3.html')
    # plot_csv_interactive(input_files=[f'{path_1}h_4.csv', f'{path_1}s_4.csv', f'{path_1}v_4.csv',f'{path_2}h_4.csv', f'{path_2}s_4.csv', f'{path_2}v_4.csv',f'{path_3}h_4.csv', f'{path_3}s_4.csv', f'{path_3}v_4.csv',f'{path_4}h_4.csv', f'{path_4}s_4.csv', f'{path_4}v_4.csv'], output_file=f'{output}compare_4.html')

    # plot_csv_interactive(input_files=[f'{path_1}h_1_smooth_4.csv', f'{path_1}s_1_smooth_4.csv', f'{path_1}v_1_smooth_4.csv',f'{path_2}h_1_smooth_4.csv', f'{path_2}s_1_smooth_4.csv', f'{path_2}v_1_smooth_4.csv',f'{path_3}h_1_smooth_4.csv', f'{path_3}s_1_smooth_4.csv', f'{path_3}v_1_smooth_4.csv',f'{path_4}h_1_smooth_4.csv', f'{path_4}s_1_smooth_4.csv', f'{path_4}v_1_smooth_4.csv'], output_file=f'{output}compare_1_smooth_4.html')
    # plot_csv_interactive(input_files=[f'{path_1}h_2_smooth_4.csv', f'{path_1}s_2_smooth_4.csv', f'{path_1}v_2_smooth_4.csv',f'{path_2}h_2_smooth_4.csv', f'{path_2}s_2_smooth_4.csv', f'{path_2}v_2_smooth_4.csv',f'{path_3}h_2_smooth_4.csv', f'{path_3}s_2_smooth_4.csv', f'{path_3}v_2_smooth_4.csv',f'{path_4}h_2_smooth_4.csv', f'{path_4}s_2_smooth_4.csv', f'{path_4}v_2_smooth_4.csv'], output_file=f'{output}compare_2_smooth_4.html')
    # plot_csv_interactive(input_files=[f'{path_1}h_3_smooth_4.csv', f'{path_1}s_3_smooth_4.csv', f'{path_1}v_3_smooth_4.csv',f'{path_2}h_3_smooth_4.csv', f'{path_2}s_3_smooth_4.csv', f'{path_2}v_3_smooth_4.csv',f'{path_3}h_3_smooth_4.csv', f'{path_3}s_3_smooth_4.csv', f'{path_3}v_3_smooth_4.csv',f'{path_4}h_3_smooth_4.csv', f'{path_4}s_3_smooth_4.csv', f'{path_4}v_3_smooth_4.csv'], output_file=f'{output}compare_3_smooth_4.html')
    # plot_csv_interactive(input_files=[f'{path_1}h_4_smooth_4.csv', f'{path_1}s_4_smooth_4.csv', f'{path_1}v_4_smooth_4.csv',f'{path_2}h_4_smooth_4.csv', f'{path_2}s_4_smooth_4.csv', f'{path_2}v_4_smooth_4.csv',f'{path_3}h_4_smooth_4.csv', f'{path_3}s_4_smooth_4.csv', f'{path_3}v_4_smooth_4.csv',f'{path_4}h_4_smooth_4.csv', f'{path_4}s_4_smooth_4.csv', f'{path_4}v_4_smooth_4.csv'], output_file=f'{output}compare_4_smooth_4.html')


    # plot_csv_interactive(input_files=[f'{path_1}h_1_smooth_10.csv', f'{path_1}s_1_smooth_10.csv', f'{path_1}v_1_smooth_10.csv',f'{path_2}h_1_smooth_10.csv', f'{path_2}s_1_smooth_10.csv', f'{path_2}v_1_smooth_10.csv',f'{path_3}h_1_smooth_10.csv', f'{path_3}s_1_smooth_10.csv', f'{path_3}v_1_smooth_10.csv',f'{path_4}h_1_smooth_10.csv', f'{path_4}s_1_smooth_10.csv', f'{path_4}v_1_smooth_10.csv'], output_file=f'{output}compare_1_smooth_10.html')
    # plot_csv_interactive(input_files=[f'{path_1}h_2_smooth_10.csv', f'{path_1}s_2_smooth_10.csv', f'{path_1}v_2_smooth_10.csv',f'{path_2}h_2_smooth_10.csv', f'{path_2}s_2_smooth_10.csv', f'{path_2}v_2_smooth_10.csv',f'{path_3}h_2_smooth_10.csv', f'{path_3}s_2_smooth_10.csv', f'{path_3}v_2_smooth_10.csv',f'{path_4}h_2_smooth_10.csv', f'{path_4}s_2_smooth_10.csv', f'{path_4}v_2_smooth_10.csv'], output_file=f'{output}compare_2_smooth_10.html')
    # plot_csv_interactive(input_files=[f'{path_1}h_3_smooth_10.csv', f'{path_1}s_3_smooth_10.csv', f'{path_1}v_3_smooth_10.csv',f'{path_2}h_3_smooth_10.csv', f'{path_2}s_3_smooth_10.csv', f'{path_2}v_3_smooth_10.csv',f'{path_3}h_3_smooth_10.csv', f'{path_3}s_3_smooth_10.csv', f'{path_3}v_3_smooth_10.csv',f'{path_4}h_3_smooth_10.csv', f'{path_4}s_3_smooth_10.csv', f'{path_4}v_3_smooth_10.csv'], output_file=f'{output}compare_3_smooth_10.html')
    # plot_csv_interactive(input_files=[f'{path_1}h_4_smooth_10.csv', f'{path_1}s_4_smooth_10.csv', f'{path_1}v_4_smooth_10.csv',f'{path_2}h_4_smooth_10.csv', f'{path_2}s_4_smooth_10.csv', f'{path_2}v_4_smooth_10.csv',f'{path_3}h_4_smooth_10.csv', f'{path_3}s_4_smooth_10.csv', f'{path_3}v_4_smooth_10.csv',f'{path_4}h_4_smooth_10.csv', f'{path_4}s_4_smooth_10.csv', f'{path_4}v_4_smooth_10.csv'], output_file=f'{output}compare_4_smooth_10.html')
    # THIS IS NO 640
    # plot_csv_interactive(input_files=[f'{path_1}h_1.csv', f'{path_1}s_1.csv', f'{path_1}v_1.csv',f'{path_2}h_1.csv', f'{path_2}s_1.csv', f'{path_2}v_1.csv',f'{path_3}h_1.csv', f'{path_3}s_1.csv', f'{path_3}v_1.csv'], output_file=f'{output}compare_1.html')
    # plot_csv_interactive(input_files=[f'{path_1}h_2.csv', f'{path_1}s_2.csv', f'{path_1}v_2.csv',f'{path_2}h_2.csv', f'{path_2}s_2.csv', f'{path_2}v_2.csv',f'{path_3}h_2.csv', f'{path_3}s_2.csv', f'{path_3}v_2.csv'], output_file=f'{output}compare_2.html')
    # plot_csv_interactive(input_files=[f'{path_1}h_3.csv', f'{path_1}s_3.csv', f'{path_1}v_3.csv',f'{path_2}h_3.csv', f'{path_2}s_3.csv', f'{path_2}v_3.csv',f'{path_3}h_3.csv', f'{path_3}s_3.csv', f'{path_3}v_3.csv'], output_file=f'{output}compare_3.html')
    # plot_csv_interactive(input_files=[f'{path_1}h_4.csv', f'{path_1}s_4.csv', f'{path_1}v_4.csv',f'{path_2}h_4.csv', f'{path_2}s_4.csv', f'{path_2}v_4.csv',f'{path_3}h_4.csv', f'{path_3}s_4.csv', f'{path_3}v_4.csv'], output_file=f'{output}compare_4.html')

    # plot_csv_interactive(input_files=[f'{path_1}h_1_smooth_4.csv', f'{path_1}s_1_smooth_4.csv', f'{path_1}v_1_smooth_4.csv',f'{path_2}h_1_smooth_4.csv', f'{path_2}s_1_smooth_4.csv', f'{path_2}v_1_smooth_4.csv',f'{path_3}h_1_smooth_4.csv', f'{path_3}s_1_smooth_4.csv', f'{path_3}v_1_smooth_4.csv'], output_file=f'{output}compare_1_smooth_4.html')
    # plot_csv_interactive(input_files=[f'{path_1}h_2_smooth_4.csv', f'{path_1}s_2_smooth_4.csv', f'{path_1}v_2_smooth_4.csv',f'{path_2}h_2_smooth_4.csv', f'{path_2}s_2_smooth_4.csv', f'{path_2}v_2_smooth_4.csv',f'{path_3}h_2_smooth_4.csv', f'{path_3}s_2_smooth_4.csv', f'{path_3}v_2_smooth_4.csv'], output_file=f'{output}compare_2_smooth_4.html')
    # plot_csv_interactive(input_files=[f'{path_1}h_3_smooth_4.csv', f'{path_1}s_3_smooth_4.csv', f'{path_1}v_3_smooth_4.csv',f'{path_2}h_3_smooth_4.csv', f'{path_2}s_3_smooth_4.csv', f'{path_2}v_3_smooth_4.csv',f'{path_3}h_3_smooth_4.csv', f'{path_3}s_3_smooth_4.csv', f'{path_3}v_3_smooth_4.csv'], output_file=f'{output}compare_3_smooth_4.html')
    # plot_csv_interactive(input_files=[f'{path_1}h_4_smooth_4.csv', f'{path_1}s_4_smooth_4.csv', f'{path_1}v_4_smooth_4.csv',f'{path_2}h_4_smooth_4.csv', f'{path_2}s_4_smooth_4.csv', f'{path_2}v_4_smooth_4.csv',f'{path_3}h_4_smooth_4.csv', f'{path_3}s_4_smooth_4.csv', f'{path_3}v_4_smooth_4.csv'], output_file=f'{output}compare_4_smooth_4.html')


    # plot_csv_interactive(input_files=[f'{path_1}h_1_smooth_10.csv', f'{path_1}s_1_smooth_10.csv', f'{path_1}v_1_smooth_10.csv',f'{path_2}h_1_smooth_10.csv', f'{path_2}s_1_smooth_10.csv', f'{path_2}v_1_smooth_10.csv',f'{path_3}h_1_smooth_10.csv', f'{path_3}s_1_smooth_10.csv', f'{path_3}v_1_smooth_10.csv'], output_file=f'{output}compare_1_smooth_10.html')
    # plot_csv_interactive(input_files=[f'{path_1}h_2_smooth_10.csv', f'{path_1}s_2_smooth_10.csv', f'{path_1}v_2_smooth_10.csv',f'{path_2}h_2_smooth_10.csv', f'{path_2}s_2_smooth_10.csv', f'{path_2}v_2_smooth_10.csv',f'{path_3}h_2_smooth_10.csv', f'{path_3}s_2_smooth_10.csv', f'{path_3}v_2_smooth_10.csv'], output_file=f'{output}compare_2_smooth_10.html')
    # plot_csv_interactive(input_files=[f'{path_1}h_3_smooth_10.csv', f'{path_1}s_3_smooth_10.csv', f'{path_1}v_3_smooth_10.csv',f'{path_2}h_3_smooth_10.csv', f'{path_2}s_3_smooth_10.csv', f'{path_2}v_3_smooth_10.csv',f'{path_3}h_3_smooth_10.csv', f'{path_3}s_3_smooth_10.csv', f'{path_3}v_3_smooth_10.csv'], output_file=f'{output}compare_3_smooth_10.html')
    # plot_csv_interactive(input_files=[f'{path_1}h_4_smooth_10.csv', f'{path_1}s_4_smooth_10.csv', f'{path_1}v_4_smooth_10.csv',f'{path_2}h_4_smooth_10.csv', f'{path_2}s_4_smooth_10.csv', f'{path_2}v_4_smooth_10.csv',f'{path_3}h_4_smooth_10.csv', f'{path_3}s_4_smooth_10.csv', f'{path_3}v_4_smooth_10.csv'], output_file=f'{output}compare_4_smooth_10.html')
    cluster([f'{path_1}h_4_smooth_10.csv', f'{path_1}s_4_smooth_10.csv'],f'./{film}/pix/',film, ks[n])
