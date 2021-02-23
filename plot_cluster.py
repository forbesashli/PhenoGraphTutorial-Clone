# %% [markdown]
# Importing the proper packages

# %%
# !pip install PhenoGraph
# !pip install opencv-contrib-python
# !pip install pyclustertend
# !pip install pytictoc
# !pip install statannot
# !pip install ffmpeg-python
# !pip install pytransform3d

# %%
import os
# Check if current environment is google colab.
# If so, execute following specific lines

data_root = os.path.join('..','PhenoGraphTutorial')
import sys
sys.path.append("..")
import numpy as np
import cv2
from skimage.transform import resize
from scipy import ndimage, misc
from scipy.ndimage import gaussian_filter
from scipy import stats,signal,fft
from scipy import io as sio
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.dates import DateFormatter
import matplotlib.patches as patches
import matplotlib.animation as manimation
from matplotlib.backends.backend_pdf import PdfPages
from mpl_toolkits import mplot3d
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd
import phenograph
import io
import imageio
from IPython.core.debugger import set_trace
from pytictoc import TicToc
from statannot import add_stat_annotation
from pathlib import Path
from numpy import sin, linspace, pi
from pylab import plot, show, title, xlabel, ylabel, subplot
import ffmpeg
from datetime import datetime
import glob
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import multiprocessing as mp
import seaborn as sns
from scipy.spatial import distance
from sklearn import metrics
from pyclustertend import hopkins
from sklearn.preprocessing import scale
from scipy.ndimage import gaussian_filter,gaussian_filter1d
from rigid_transform_3D import rigid_transform_3D
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram
from pytransform3d.rotations import *
# %%
# !pip install ludwig

# %%
import matplotlib
matplotlib.use('tkagg')


# %% [markdown]
# Here is the definitions

# %%
#@title function which returns an image as numpy array from figure
def get_img_from_fig(fig, dpi=180):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi)
    buf.seek(0)
    img_arr = np.frombuffer(buf.getvalue(), dtype=np.uint8)
    buf.close()
    img = cv2.imdecode(img_arr, 1)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    return img

def plot_dendrogram(model, **kwargs):
    # Create linkage matrix and then plot the dendrogram
    # create the counts of samples under each node
    counts = np.zeros(model.children_.shape[0])
    n_samples = len(model.labels_)
    for i, merge in enumerate(model.children_):
        current_count = 0
        for child_idx in merge:
            if child_idx < n_samples:
                current_count += 1  # leaf node
            else:
                current_count += counts[child_idx - n_samples]
        counts[i] = current_count
    linkage_matrix = np.column_stack([model.children_, model.distances_,
                                      counts]).astype(float)
    # Plot the corresponding dendrogram
    dendrogram(linkage_matrix, **kwargs)
# %% [markdown]
# Data matrixes

# %%
#mouse wheel joints
joints = [[0,1],[1,2],[2,3],[3,4],[4,5],[5,6],[6,7],[7,8],[8,9],[9,10],
          [10,11],[11,12],[12,13],[14,15],[16,17],[17,18],[18,19],[19,20],
          [20,21],[22,23],[23,24],[24,25],[25,26],[26,27]]

# data_2d = ['LD1_1580414966_2d.csv', 'LD1_1580415036_2d.csv', 'LD1_1580415176_2d.csv',
#            'LD1_1580415664_2d.csv', 'LD1_1580416013_2d.csv', 'LD1_1580416083_2d.csv',
#            'LD1_1580416431_2d.csv', 'LD1_1580416571_2d.csv', 'LD1_1580416920_2d.csv',
#            'LD1_1580417059_2d.csv', 'LD1_1580417618_2d.csv', 'LD1_1580417687_2d.csv',
#            'LD1_1580418315_2d.csv', 'LD1_1580418873_2d.csv', 'LD1_1580419640_2d.csv']
#
# data_3d = ['LD1_1580414966_3d.csv', 'LD1_1580415036_3d.csv', 'LD1_1580415176_3d.csv',
#            'LD1_1580415664_3d.csv', 'LD1_1580416013_3d.csv', 'LD1_1580416083_3d.csv',
#            'LD1_1580416431_3d.csv', 'LD1_1580416571_3d.csv', 'LD1_1580416920_3d.csv',
#            'LD1_1580417059_3d.csv', 'LD1_1580417618_3d.csv', 'LD1_1580417687_3d.csv',
#            'LD1_1580418315_3d.csv', 'LD1_1580418873_3d.csv', 'LD1_1580419640_3d.csv']
# data_2d = ['LD1_1580414966_2d.csv',
#           'LD1_1580415036_2d.csv', 'LD1_1580415176_2d.csv',
#           'LD1_1580416083_2d.csv', 'LD1_1580416920_2d.csv'
#           ]

data_2d = [
           'LD1_1580415036_2d.csv'
           ]

# data_3d = ['LD1_1580414966_3d.csv',
#           'LD1_1580415036_3d.csv', 'LD1_1580415176_3d.csv',
#           'LD1_1580416083_3d.csv', 'LD1_1580416920_3d.csv'
#           ]
data_3d = [
           'LD1_1580415036_3d.csv'
           ]

coords_all_2d = []
coords_all_3d = []
dataset_name_2d = []
dataset_name_3d = []

for f_2d, f_3d in zip(data_2d,data_3d):
    coords_file = data_root + os.sep + f_2d
    dataset_name_2d = coords_file.split('/')[-1].split('.')[0]
    coords_2d = pd.read_csv(coords_file,dtype=np.float, header=2)
    coords_2d = coords_2d.values[:,1:] #exclude first column
    coords_2d = np.delete(coords_2d, list(range(2, coords_2d.shape[1], 3)), axis=1) #delete every 3rd column of prediction score
    coords_all_2d.append(coords_2d)

    coords_file = data_root + os.sep + f_3d
    dataset_name_3d = coords_file.split('/')[-1].split('.')[0]
    coords_3d = pd.read_csv(coords_file, header=2)
    coords_3d = coords_3d.values[:, 1:] #exclude the index column
    coords_3d = np.around(coords_3d.astype('float'), 2) #round to two decimal places
    coords_3d = gaussian_filter1d(coords_3d, 5, axis=0) #smooth the data, the points were oscillating
    coords_all_3d.append(coords_3d)

# %%

coords_all_2d = np.vstack(coords_all_2d) #convert to numpy stacked array
coords_all_3d = np.vstack(coords_all_3d)
x_3d = coords_all_3d[:, ::3];        y_3d = coords_all_3d[:, 1::3];        z_3d = coords_all_3d[:, 2::3];
x_2d = coords_all_2d[:, ::2];        y_2d = coords_all_2d[:, 1::2];        z_2d = np.zeros(x_2d.shape);

k=30 # K for k-means step of phenograph
communities_2d, graph, Q = phenograph.cluster(coords_all_2d, k=k)
n_clus_2d = np.unique(communities_2d).shape[0]
communities_3d, graph, Q = phenograph.cluster(coords_all_3d, k=k)
n_clus_3d = np.unique(communities_3d).shape[0]

hopkins(coords_all_2d, coords_all_2d.shape[0])
hopkins(coords_all_3d, coords_all_3d.shape[0])

# %% tSNE plot of 2d data labeled by phenograph clusters
if not os.path.exists('syn_2d_tsne.png'):
    # tsne_model = TSNE(n_components=2, perplexity=30.0, early_exaggeration=12.0,
    #                   learning_rate=20.0, n_iter=1000, n_iter_without_progress=300,
    #                   min_grad_norm=1e-07, metric='euclidean', init='random',
    #                   verbose=0, random_state=None, method='barnes_hut',
    #                   angle=0.5)
    # tsne_model = TSNE(n_components=2,perplexity=30,random_state=0)
    tsne_model = TSNE(n_components=2, random_state=2,perplexity=100,angle=0.1,init='pca',n_jobs= mp.cpu_count()-1)
    Y_2d = tsne_model.fit_transform(coords_all_2d)
    # cmap = matplotlib.colors.ListedColormap ( np.random.rand ( np.unique(communities_re).shape[0],3))
    cmap = plt.cm.colors.ListedColormap(plt.cm.jet(np.linspace(0,1,n_clus_2d)))
    plt.figure()
    plt.scatter(Y_2d[:,0], Y_2d[:,1],
                    c=communities_2d,
                    cmap=cmap,
                    alpha=1.0)
    plt.colorbar(ticks=np.unique(communities_2d), label='Cluster#')
    plt.xlabel('TSNE1'); plt.ylabel('TSNE2')
    plt.title('2D Body coordinate clusters: total frames ' + str(len(communities_2d)))
    plt.savefig('syn_2d_tsne.png', format='png')
    plt.show(block=False)
    plt.close()

# %%
tsne_model = TSNE(n_components=2, random_state=2,perplexity=100,angle=0.1,init='pca',n_jobs= mp.cpu_count()-1)

# %% tSNE plot of 3d data labeled by phenograph clusters
if not os.path.exists('syn_3d_tsne.png'):
    # tsne_model = TSNE(n_components=2, perplexity=30.0, early_exaggeration=12.0,
    #                   learning_rate=20.0, n_iter=1000, n_iter_without_progress=300,
    #                   min_grad_norm=1e-07, metric='euclidean', init='random',
    #                   verbose=0, random_state=None, method='barnes_hut',
    #                   angle=0.5)
    # tsne_model = TSNE(n_components=2,perplexity=30,random_state=0)
    tsne_model = TSNE(n_components=2, random_state=2,perplexity=100,angle=0.1,init='pca',n_jobs= mp.cpu_count()-1)
    Y = tsne_model.fit_transform(coords_all_3d)
    # cmap = matplotlib.colors.ListedColormap ( np.random.rand ( np.unique(communities_sy).shape[0],3))
    cmap = plt.cm.colors.ListedColormap(plt.cm.jet(np.linspace(0,1,n_clus_3d)))
    plt.figure()
    plt.scatter(Y[:,0], Y[:,1],
                c=communities_3d,
                cmap=cmap,
                alpha=1.0)
    plt.colorbar(ticks=np.unique(communities_3d), label='Cluster#')
    plt.xlabel('TSNE1'); plt.ylabel('TSNE2')
    plt.title('3D Body coordinate clusters: total frames ' + str(len(communities_3d)))
    plt.savefig('syn_3d_tsne.png', format='png')
    plt.show(block=False)
    plt.close()

