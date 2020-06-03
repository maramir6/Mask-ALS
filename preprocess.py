from laspy.file import File
from laspy.header import Header
import numpy as np
from numba import jit
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from skimage.filters.rank import otsu
from skimage.morphology import disk, dilation, remove_small_holes, binary_dilation, binary_erosion
from skimage.filters import threshold_otsu, scharr
from skimage.filters.rank import equalize, mean, median
from skimage.measure import label, regionprops_table
from skimage.feature import peak_local_max
from skimage.transform import resize
from skimage.segmentation import watershed
from skimage.color import gray2rgb, rgba2rgb, label2rgb
from skimage.exposure import rescale_intensity, equalize_hist, equalize_adapthist
from skimage.io import imsave
from skimage.draw import circle_perimeter, circle
import os, sys, cv2
from osgeo import gdal, gdalnumeric, ogr, osr
import pandas as pd
from scipy.stats import mode
from scipy import ndimage, misc
import cmapy

@jit(nopython=True, cache=True)
def raster_density(y, x, scale):
    
    factor, n_points = (1/scale), len(y)
    
    min_y, min_x = np.min(y), np.min(x)
    max_y, max_x = np.max(y), np.max(x)    
    yi, xi = (factor*(max_y -y)).astype(np.int32), (factor*(x - min_x)).astype(np.int32)
    
    n_rows, n_columns = int(np.max(yi) + 1), int(np.max(xi) + 1)
    raster = np.full((n_rows, n_columns), 0, dtype=np.int32)
    
    for i in range(n_points):
        raster[yi[i],xi[i]] = raster[yi[i],xi[i]] + 1  
        
    return raster

@jit(nopython=True, cache=True)
def raster_weight(y, x, z, scale):
    
    factor, n_points = (1/scale), len(y)
    
    min_y, min_x = np.min(y), np.min(x)
    max_y, max_x = np.max(y), np.max(x)    
    max_z = np.max(z)
    
    yi, xi = (factor*(max_y -y)).astype(np.int32), (factor*(x - min_x)).astype(np.int32)
    
    n_rows, n_columns = int(np.max(yi) + 1), int(np.max(xi) + 1)
    raster = np.full((n_rows, n_columns), 0, dtype=np.float32)
    
    for i in range(n_points):
        raster[yi[i],xi[i]] = raster[yi[i],xi[i]] + (z[i]/max_z)  
        
    return raster

@jit(nopython=True, cache=True)
def raster_height(y, x, z, scale):
    
    factor, n_points = (1/scale), len(y)
    
    min_y, min_x = np.min(y), np.min(x)
    max_y, max_x = np.max(y), np.max(x)
    yi, xi = (factor*(max_y -y)).astype(np.int32), (factor*(x - min_x)).astype(np.int32)
    
    n_rows, n_columns = int(np.max(yi) + 1), int(np.max(xi) + 1)
    raster = np.full((n_rows, n_columns), 0, dtype=np.float32)
    
    for i in range(n_points):
        z_i = raster[yi[i],xi[i]]
        
        if z[i] > z_i:
            raster[yi[i],xi[i]] = z[i]
        
    return raster

def merge_areas(labels):
    
    props = regionprops_table(labels, properties=('label', 'area', 'major_axis_length', 'minor_axis_length')) 
    df = pd.DataFrame(props)
    df['coef'] = df['major_axis_length']/df['minor_axis_length']
    area = df['area'].values
    coef = df['coef'].values
    unique = df['label'].values

    min_area = np.percentile(area, 2)
    
    print('Minimum area: ', min_area, 'and trees detected are ', len(df.index))

    indice = np.where((coef>3)|(area<min_area))
    s_label = unique[indice]
    
    for i in range(len(s_label)):
        mask = labels==s_label[i]
        mask_d = binary_dilation(mask)
        mask_c = np.logical_xor(mask_d, mask)
        label = mode(np.ravel(labels[mask_c])).mode
        labels[mask] = label

    return labels

def files_list(folder_path):
    return [f for f in os.listdir(folder_path) if f.endswith('.laz')]

def draw_als(event,x,y,flags,param):
    global ix,iy,drawing,mode

    if event == cv2.EVENT_LBUTTONUP:
        cv2.circle(dimage,(x,y),2,(0,0,255),-1)
        markers_w[(y-1):(y+1),(x-1):(x+1)] = True

    elif event == cv2.EVENT_RBUTTONUP:
        cv2.circle(dimage,(x,y),2,(230,230,230),-1)
        markers_w[(y-4):(y+4),(x-4):(x+4)] = False

def drawing_windows(dimage):

    cv2.namedWindow('als', cv2.WINDOW_NORMAL)
    cv2.setMouseCallback('als', draw_als)
    
    while(1):
        cv2.imshow('als',dimage)
        k = cv2.waitKey(1) & 0xFF

        if k == ord('m'):
            mode = not mode

        elif k == ord('y'):
            break
    
    cv2.destroyAllWindows()

def check_windows(labels, map_h):

    mask = labels > 0
    mask = np.stack((mask, mask, mask), axis=2)

    labels_color = label2rgb(labels)
    labels_color = np.multiply(labels_color, mask)
    image = np.hstack((255*labels_color, map_h)).astype(np.uint8)

    cv2.namedWindow('als', cv2.WINDOW_NORMAL)
    print('Yes(Y) or no(N) ....')

    while(1):

        cv2.imshow('als', image)
        k = cv2.waitKey(1) & 0xFF

        if k == ord('y'):
            save = True
            break

        elif k == ord('n'):
            save = False
            break
    
    cv2.destroyAllWindows()

    return save

def image_augmentation(image, size, outfolder, file):

    hsize =(size/2)
    x_n, y_n, z_n = image.shape
    xi, yi = x_n//hsize, y_n//hsize

    for i in range(int(xi-1)):
        for j in range(int(yi-1)):
            
            crop = image[int(i*hsize):int(i*hsize+size),int(j*hsize):int(j*hsize+size), :]
            imsave(outfolder + file + '_1.png', crop)
            
            crop_90 = ndimage.rotate(crop, 90)
            imsave(outfolder + file + '_2.png', crop_90)

            crop_180 = ndimage.rotate(crop, 180)
            imsave(outfolder + file + '_3.png', crop_180)

            crop_270 = ndimage.rotate(crop, 270)
            imsave(outfolder + file + '_4.png', crop_270)

def labels_augmentation(image, size, outfolder, file):

    hsize =(size/2)
    x_n, y_n = image.shape
    xi, yi = x_n//hsize, y_n//hsize

    for i in range(int(xi-1)):
        for j in range(int(yi-1)):
            
            crop = image[int(i*hsize):int(i*hsize+size),int(j*hsize):int(j*hsize+size)]
            imsave(outfolder + file + '_1.png', crop)
            
            crop_90 = ndimage.rotate(crop, 90)
            imsave(outfolder + file + '_2.png', crop_90)
            
            crop_180 = ndimage.rotate(crop, 180)
            imsave(outfolder + file + '_3.png', crop_180)
            
            crop_270 = ndimage.rotate(crop, 270)
            imsave(outfolder + file + '_4.png', crop_270)

@jit(nopython=True, cache=True)
def knndistance(lats, lons, k):

    n_points = len(lats)
    nbour_points = np.full(n_points, np.nan, dtype=np.float32)
    valid_points = np.full(n_points, True)
    
    for i in range(n_points):
        lat, lon = lats[i], lons[i]
        dist = np.sqrt(np.power(lats-lat,2)+np.power(lons-lon,2))

        i_p = np.argsort(dist)[1:(k+1)]
        d_p = np.median(dist[i_p])

        nbour_points[i] = d_p

    return nbour_points

@jit(nopython=True, cache=True)
def max_height_radii(h_map, lats, lons, px_radii):
    
    n_points = len(lats)
    y,x = np.ogrid[-a:px_radii-a, -b:px_radii-b]
    mask = x*x + y*y <= px_radii*px_radii

    for i in range(n_points):
        lat, lon = lats[i], lons[i]
        m_y, m_x = np.argmax(h_map[(lat-px_radii):(lat+px_radii), (lon-px_radii):(lon+px_radii)]*mask)

def process_file(folder, file, outfolder):
    # Load LiDar filex and read X,Y,Z
    inFile = File(folder + file, mode = "r")
    header = inFile.header

    X, Y, Z = inFile.x, inFile.y, inFile.z

    scale_d, scale_h = 0.5, 0.15
    dist_min, perc_factor = 1.2, 75
    n_pad, size_img = 5, 1200
    
    # Density and canopy height map creation
    map_h = raster_height(Y, X, Z, scale_h)
    map_w = raster_weight(Y, X, Z, scale_d)
    
    # Resize density map
    map_w = resize(map_w, map_h.shape)

    # Calculate center of image and radii in pixels
    y_n, x_n = map_w.shape
    y_c, x_c = int(0.5*y_n), int(0.5*x_n)
    radii = min(y_c, x_c)

    # Mask creation
    mask = np.zeros((y_n, x_n), dtype=np.uint8)
    rr, cc = circle(y_c, x_c, radii)
    mask[rr, cc] = 1

    # Padding
    mask = np.pad(mask, [(n_pad, n_pad), (n_pad, n_pad)], mode='constant')
    map_h = np.pad(map_h, [(n_pad, n_pad), (n_pad, n_pad)], mode='constant')
    map_w = np.pad(map_w, [(n_pad, n_pad), (n_pad, n_pad)], mode='constant')
    
    # Threshold Otsu
    h_min = 0.85*threshold_otsu(map_h)
    h_mask = map_h>h_min
    
    map_h = np.multiply(h_mask, map_h)
    score = np.multiply(h_mask, map_w)
    
    threshold_w = np.percentile(map_w, perc_factor)

    # Max calculation
    markers = peak_local_max(score, indices=False, min_distance=int(dist_min/scale_h), exclude_border=True, threshold_abs= threshold_w, labels=binary_erosion(mask, disk(2)))

    # Global variables

    lats, lons = np.where(markers == 1)
    dist = knndistance(lats, lons, 1)

    print('Median distance between trees: ', scale_h*np.median(dist))

    markers_w = dilation(markers, disk(1)).astype(int)

    global dimage

    # Normalizar height map
    max_h = np.max(map_h)
    map_h = map_h/max_h

    
    map_h = median(map_h/np.max(map_h), disk(1))
    dimage = np.clip(gray2rgb(map_h) + np.stack((-255*markers_w, -255*markers_w, 225*markers_w), axis=2), a_min = 0, a_max = 255).astype(np.uint8)

    # Watershed segmentation
    markers_w = dilation(markers, disk(3))
    markers_w = label(markers_w)

    mask = np.multiply(h_mask, mask)
    labels = watershed(-map_h, markers_w, mask=mask)

    labels = merge_areas(labels)

    map_h = gray2rgb(map_h).astype(np.uint8)
    
    save = check_windows(labels, dimage)

    if save:
        if np.random.rand(1) < 0.85:
            outfold = outfolder + 'TRAIN/'
        else:
            outfold = outfolder + 'TEST/'

        image_augmentation(map_h, 256, outfold + 'IMAGES/', file[:-4])
        labels_augmentation(labels, 256, outfold + 'LABELS/', file[:-4])


    imsave(folder + file[:-4] + '_h.png', map_h)
    mpimg.imsave(folder + file[:-4] + '_l.png', labels, cmap=plt.cm.nipy_spectral)
    
    print(file[:-4] + ' saved.')
    
    inFile.close()

    return index_img, index_lb

path = 'E:/RODALES/PARCELA_COSECHA/'
outfolder = 'E:/RODALES/PARCELA_COSECHA/'

index_img, index_lb = 0, 0

for file in files_list(path):
    print('Preprocess: ', file)
    index_img, index_lb = process_file(path, file, outfolder)



