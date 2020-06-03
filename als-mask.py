from laspy.file import File
from laspy.header import Header
import numpy as np
import pandas as pd
import geopandas as gpd
import fiona
from numba import jit
from shapely.geometry import Polygon
import matplotlib.pyplot as plt
from skimage.filters.rank import otsu, median
from skimage.filters import threshold_otsu, scharr
from skimage.morphology import disk, dilation, remove_small_holes
from skimage.measure import label
from skimage.feature import peak_local_max
from skimage.draw import circle_perimeter, circle
import os, sys, copy, time, cv2
from scipy import ndimage, stats
from osgeo import gdal, gdalnumeric, ogr, osr
import matplotlib.image as mpimg
import dask.dataframe as dd
from dask import delayed, compute
from scipy.spatial import distance
from PIL import Image, ImageDraw
import json, argparse
from scipy.stats import kurtosis, skew
from dataset_als import LidarDataset
import warnings

import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from torchvision.models.detection.rpn import AnchorGenerator
from engine import train_one_epoch, evaluate
import utils
import transforms as T
import torch

def get_model_instance_segmentation(num_classes, input_channel, hidden_layer=256):
    
    # load an instance segmentation model pre-trained pre-trained on COCO
    model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=False)
    #model.rpn.anchor_generator = AnchorGenerator(sizes=((16, 32, 64, 128),), aspect_ratios=((0.5, 1.0, 2.0),))
    
    # get number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    
    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    # now get the number of input features for the mask classifier
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels

    # and replace the mask predictor with a new one
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask, hidden_layer, num_classes)

    return model

warnings.filterwarnings("ignore")

def vec_translate(a, my_dict):
    return np.vectorize(my_dict.__getitem__)(a)

def kantor_encoder(y, x):
    return 0.5*(y+x)*(y+x+1)+y

def create_folder(out_path):
    if not os.path.exists(out_path):
        os.makedirs(out_path)

def files_list(folder_path):

    return [f for f in os.listdir(folder_path) if f.endswith('.laz')]

def array_constructor(X, Y, Z, L):
    indices = np.argsort(L)
    X, Y, Z = X[indices], Y[indices], Z[indices]
    unique, counts = np.unique(L, return_counts=True)
    array = loop_constructor(unique, counts, X, Y, Z)
    return array

@jit(nopython=True, cache=True)
def metrics_computation(counts, X, Y, Z):

    n_tree = len(counts)
    metrics = np.full(n_tree, np.nan, dtype = np.double)

    for u_id in range(0, n_tree):

        c_id = counts[u_id]
        Xt, Yt, Zt = X[:c_id], Y[:c_id], Z[:c_id]

        indice = Zt > 4
        Xt, Yt, Zt = Xt[indice], Yt[indice], Zt[indice]
        prun  = np.nan

        if len(Xt) > 10:

            prun = 0.0

            indice = Zt <= 5.5
            Xp, Yp = Xt[indice], Yt[indice]
            Xc, Yc = Xt[~indice], Yt[~indice]
        
            if (len(Xp)<2):
                prun = 1.0
            else:
                dif_p = 0.5*(np.max(Xp) - np.min(Xp)) + 0.5*(np.max(Yp) - np.min(Yp))
                dif_c = 0.5*(np.max(Xc) - np.min(Xc)) + 0.5*(np.max(Yc) - np.min(Yc))

                if(dif_p<0.26*dif_c):
                    prun = 1.0

            if prun==1.0:

                dif_p = 2*dif_p

                sorted_indice = np.argsort(Zt)
                Xt, Yt, Zt = Xt[sorted_indice], Yt[sorted_indice], Zt[sorted_indice]
                
                bins_count, bins_edge = np.histogram(Zt, bins=23, range=(2.7, 8.5))

                for id_bin in range(len(bins_count)):
                    
                    c_bin = bins_count[id_bin]

                    if c_bin > 0:
                        
                        Xb, Yb, Zb = Xt[:c_bin], Yt[:c_bin], Zt[:c_bin]
                        dif_b = 0.5*(np.max(Xb) - np.min(Xb)) + 0.5*(np.max(Yb) - np.min(Yb))

                        if dif_b >= dif_p:
                            prun = np.around(np.min(Zb), 1)
                            break 

                        Xt, Yt, Zt = Xt[c_bin:], Yt[c_bin:], Zt[c_bin:]

            metrics[u_id] = prun

        X, Y, Z = X[c_id:], Y[c_id:], Z[c_id:]

    return metrics

def trees_points(X, Y, Z, L, folder, header, save = False):

    indices = np.argsort(L)
    X, Y, Z = X[indices], Y[indices], Z[indices]

    max_bin = np.max(L) + 1 
    count, bins_edge = np.histogram(L, bins=max_bin, range=(0, max_bin))

    cg = count[0]
    X, Y, Z = X[cg:], Y[cg:], Z[cg:]
    count = count[1:]
    prun_h = metrics_computation(count, X, Y, Z)

    # Replace unknown values prun height for the mean value
    prun_mean = np.around(np.nanmean(prun_h[prun_h>1]),1)
    prun_h = np.where(prun_h==1, prun_mean, prun_h)
    
    if save:
        save_trees(count, prun_h, folder, header, X, Y, Z)
    
    return prun_h

def save_trees(counts, metrics, folder, header, X, Y, Z):

    id_tree = 0

    for u_id in range(0, len(counts)):
        c_id = counts[u_id]
        Xt, Yt, Zt = X[:c_id], Y[:c_id], Z[:c_id]
        if np.isfinite(metrics[u_id]):
            save_file(folder + str(id_tree) + '.laz', header, Xt, Yt, Zt)
            id_tree = id_tree + 1
        X, Y, Z = X[c_id:], Y[c_id:], Z[c_id:]

def tree_id(labels, X, Y):
    n_rows, n_columns = labels.shape
    y_i, x_i = np.repeat(np.arange(n_rows), n_columns), np.tile(np.arange(n_columns), n_rows)
    key_kantor, values = kantor_encoder(y_i, x_i).astype(int), np.ravel(labels).astype(int)
    dictionary = dict(zip(key_kantor.tolist(), values.tolist()))
    data_kantor = kantor_encoder(Y, X).astype(int)
    i = vec_translate(data_kantor, dictionary)

    return i

def world2Pixel(geoMatrix, x, y):
    ulX, ulY = geoMatrix[0], geoMatrix[3]
    xDist, yDist = geoMatrix[1], geoMatrix[5]

    pixel = int((x - ulX) / xDist)

@jit(nopython=True, cache=True)
def density_raster(lats, lons, boundary, factor=25, rodal_len=20):
    
    n_points = len(lats)

    max_y, min_y = boundary[0], boundary[1]
    max_x, min_x = boundary[2], boundary[3]

    geotrans = [min_x, rodal_len, 0, max_y, 0, -rodal_len]

    lats_r, lons_r = ((1/rodal_len)*(max_y - lats)).astype(np.int32), ((1/rodal_len)*(lons - min_x)).astype(np.int32)

    size_y, size_x = int(max_y/rodal_len) - int(min_y/rodal_len) + 1, int(max_x/rodal_len) - int(min_x/rodal_len) + 1

    d_raster = np.full((size_y, size_x), 0, dtype=np.int32)

    for i in range(n_points):
        lat, lon = lats_r[i], lons_r[i]
        d_raster[lat, lon] = d_raster[lat, lon] + factor

    d_rod = np.full(n_points, 0, dtype=np.int32)

    for i in range(n_points):
        lat, lon = lats_r[i], lons_r[i]
        d_rod[i] = d_raster[lat, lon]


    return d_raster, geotrans, d_rod

def save_raster(file_name, image, geotrans, data_type = gdal.GDT_UInt32):

    driver = gdal.GetDriverByName('GTiff')

    projection = osr.SpatialReference()
    projection.ImportFromEPSG(32718)

    raster = driver.Create(file_name, image.shape[1], image.shape[0], 1, data_type)
    raster.SetGeoTransform(geotrans)  
    raster.GetRasterBand(1).WriteArray(image)
    raster.SetProjection(projection.ExportToWkt())
    raster.FlushCache()


@jit(nopython=True, cache=True)
def height(data):

    max_rows = np.max(data[:,0])
    max_columns = np.max(data[:,1])
    n_rows, n_columns = int(max_rows + 1), int(max_columns + 1)
    array_max = np.full((n_rows, n_columns), 0, dtype=np.float32)
    array_den = np.full((n_rows, n_columns), 0, dtype=np.int32)
    
    for i in range(data.shape[0]):
        array_max[int(data[i][0]), int(data[i][1])] = data[i][2]
        array_den[int(data[i][0]), int(data[i][1])] = data[i][3]

    return array_max, array_den

def labels_image(X, Y, lats, lons, buffer_size):

    n_scale = 0.15
    min_x, max_y = np.min(X), np.max(Y)
    rows , columns = (1/n_scale)*(max_y - lats - 0.5*n_scale), (1/n_scale)*(lons - min_x - 0.5*n_scale)
    X = np.array((1/n_scale)*(X - min_x), dtype=np.int)
    Y = np.array((1/n_scale)*(max_y -Y), dtype=np.int)
    labels = np.full((int(np.max(Y) + 1), int(np.max(X) + 1)), 0, dtype=np.int)

    for index in range(0, len(rows)):
        circy, circx = circle(rows[index], columns[index], int(buffer_size/n_scale), shape=labels.shape)
        labels[circy, circx] = index + 1

    return labels, X, Y

def check_path(path):
    if path[-1] != '/':
        path = path + '/'

    return path

def save_file(file_name, header, x, y, z):
    with File(file_name, mode="w", header=header) as outfile:
        outfile.x, outfile.y, outfile.z = x, y, z

def write_txt(file_name, files):

    with open(file_name, 'w') as writer:

        for file in files:
            writer.write(file + '\n')

@jit(nopython=True, cache=True)
def points_array(z, count, array):
    for i in range(len(count)):
        c_i = count[i]
        array[i] = np.max(z[:c_i])
        z = z[c_i:]

    return array

def raster(X, Y, Z):

	K = ((Y+X)*(Y+X+1)+Y)/2
	unique, index,  count = np.unique(K, return_index = True, return_counts = True)

	indice = np.argsort(K)
	Z = Z[indice]

	Z_max = np.zeros(len(count)).astype(float)
	Z_max = points_array(Z, count, Z_max)

	Y, X = Y[index], X[index]

	return np.stack((Y, X, Z_max, count), axis=1)

def area_calculation(d_raster, scale):

    mask = d_raster > 0
    mask = remove_small_holes(mask, int(scale*50))
    area = (0.0001)*(scale*scale)*len(np.ravel(mask[mask==True]))

    return area

def alpha_value(dictionary):

    alpha_vol = {"ZC1":0.0002345621, "ZC2":0.0002421476, "ZC4":0.0002275834, "ZC5":0.0002134500, "ZC6":0.0002507412, "ZC7":0.0002286407, "ZC9":0.0002244830, "ZC10":0.0002226959}
    alpha_dap = {'ZC1':52.3066891710 ,   'ZC2':51.2588421817,  'ZC4':51.3986769080,  'ZC5':50.5393098756,  'ZC6':52.6842435696, 'ZC7':49.0946360378,  'ZC9':50.0895492304,  'ZC10': 50.6917291971}
    alpha_ab = { 'ZC1':0.002634596, 'ZC2':0.002570477, 'ZC4':0.002418060, 'ZC5':0.002470647, 'ZC6':0.002701253, 'ZC7':0.002390591, 'ZC9':0.002538903,  'ZC10':0.002516450}

    keys = [k for k in dictionary.keys() if (k.startswith('ZC'))&(dictionary[k]==1)]
    key = keys[0][:2] + keys[0][8:]

    return alpha_vol[key], alpha_dap[key], alpha_ab[key]

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

@jit(nopython=True, cache=True)
def drop_duplicates(lats, lons, min_d):

    n_points = len(lats)

    for i in range(n_points):

        lat, lon = lats[i], lons[i]

        for j in range(i+1, n_points):

            dist = np.sqrt(np.power((lats[j] - lat),2)+np.power((lons[j] - lon),2))

            if dist < min_d:
                lats[j], lons[j] = np.nan, np.nan
                break

    indice = np.isfinite(lats)

    return lats[indice], lons[indice]


def run(folder, file, scale, dist_min, perc_factor, h_min, dictionary, output_folder):

    # Scale parameters for voxelization(unit in meters)
    buffer_size = 0.9*dist_min

    # Load LiDar filex and read X,Y,Z
    inFile = File(folder + file, mode = "r")
    header = inFile.header

    # Create a folder to store files
    folder = output_folder + file[:-4] + '/'
    create_folder(folder)

    start_time = time.time()

    X, Y, Z = inFile.x, inFile.y, inFile.z

    # Calculate Min Max for every dimension
    max_x, min_x = np.max(X), np.min(X)
    max_y, min_y = np.max(Y), np.min(Y)
    max_z, min_z = np.max(Z), np.min(Z)

    map_h = raster_height(Y, X, Z, scale)
    n_rows, n_columns = map_h.shape

    # Threshold Otsu
    h_min = 0.85*threshold_otsu(map_h)
    h_mask = map_h>h_min
    
    map_h = np.multiply(h_mask, map_h)

    # Estimate area in ha
    area = area_calculation(map_h, scale)

    map_h = median(map_h/np.max(map_h), disk(1))
    max_h = np.max(map_h)
    map_h = map_h/max_h

    num_classes = 2
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    # get the model using our helper function
    model = get_model_instance_segmentation(num_classes, input_channel=3)
    checkpoint = torch.load('E:/PYTORCH-MASK/model_final_2.pth')
    model.load_state_dict(checkpoint)
    model.eval()
    model.to(device)

    cpu_device = torch.device("cpu")
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Prediction:'

    dataset = LidarDataset(map_h, 256)    
    inf_bnd, sup_bnd = int(0.05*256), int(0.95*256)

    data_loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=4, collate_fn=utils.collate_fn)
    rows, columns = np.array([]), np.array([])
    id_img = 0
    
    for image, coords in metric_logger.log_every(data_loader, 100, header):

        image = list(img.to(device) for img in image)
        torch.cuda.synchronize()
        outputs = model(image)

        outputs = [{k: v.to(cpu_device) for k, v in t.items()} for t in outputs]

        index = 0
        
        for output in outputs:

            boxes = output["boxes"].detach().numpy()
            y_ref, x_ref = coords[index]

            x_min, y_min, x_max, y_max = boxes[:,0], boxes[:,1], boxes[:,2], boxes[:,3]
                    
            y, x = 0.5*y_max + 0.5*y_min, 0.5*x_max + 0.5*x_min
            inbound = np.where((y<=sup_bnd)&(y>=inf_bnd)&(x<=sup_bnd)&(x>=inf_bnd))
            
            y, x = y[inbound], x[inbound]
            y, x = drop_duplicates(y, x, int(dist_min/scale))

            rows = np.append(rows, y + y_ref) 
            columns = np.append(columns, x + x_ref)

#            idx, idy = (0.5*boxes[:,3] + 0.5*boxes[:,1]).astype(int), (0.5*boxes[:,2] + 0.5*boxes[:,0]).astype(int)

#            mask = np.zeros((image_chm.shape[1], image_chm.shape[2]))
#            mask[idx, idy] = 1
            
#            dimage = np.clip(np.stack((image_chm[0,:,:], image_chm[1,:,:], image_chm[2,:,:]), axis=2) + np.stack((255*mask, -255*mask, -225*mask), axis=2), a_min = 0, a_max = 255).astype(np.uint8)
#            mpimg.imsave(folder + file[:-4] +  str(id_img) +'.png', dimage)
#            id_img = id_img + 1
            index = index + 1

    lats, lons = -scale*rows + max_y - 0.5*scale, scale*columns + min_x + 0.5*scale

    # Label trees with a circle of radii = buffer_size
    labels, Xp, Yp = labels_image(X, Y, lats, lons, buffer_size)

    i = tree_id(labels, Xp, Yp)

    metrics = trees_points(X, Y, Z, i, folder, header)

    # Filter valid trees
    valid_tree = np.isfinite(metrics)
    rows, columns = rows[valid_tree], columns[valid_tree]
    lats, lons = lats[valid_tree], lons[valid_tree]
    metrics = metrics[valid_tree]

    n_trees = len(lats)
    n_prun = len(metrics[metrics>0])

    tree_avg = int(n_trees/area)
    prun_avg = int(n_prun/area)

    print(n_trees,' trees were succesfully clipped in', "--- %s seconds ---" % (time.time() - start_time))
    print(tree_avg, ' average of trees per hectare')
    print(prun_avg, ' average of prunned trees per hectare')

    # Compute other metrics
    map_h = max_h*map_h
    index = np.arange(n_trees)
    h_v = np.ravel(map_h)
    h_h = np.ravel(map_h, order='F')

    index_v = (rows*n_columns + columns).astype(int)
    index_h = (columns*n_rows + rows).astype(int)

    h1 = np.take(h_v, index_v)
    h2 = np.take(h_v, index_v-1)
    h3 = np.take(h_h, index_h-1)
    h4 = np.take(h_v, index_v+1)
    h5 = np.take(h_h, index_h+1)

    h_max = np.stack((h1, h2, h3, h4, h5), axis=1)
    h_max = np.max(h_max, axis=1)

    # Save CHM
    boundary = (max_y, min_y, max_x, min_x)
    geotrans = [min_x, scale, 0, max_y, 0, -scale]
    save_raster(folder + file[:-4] + '_CHM.tif', map_h, geotrans, data_type = gdal.GDT_Float32)

    # Save Density Raster
    d_raster, geotrans, d_rod = density_raster(lats, lons, boundary, factor=100, rodal_len=10)
    save_raster(folder + file[:-4] + '_TD.tif', d_raster, geotrans, data_type = gdal.GDT_UInt32)

    # Save Prun Density Raster
    p_raster, geotrans, p_rod = density_raster(lats[metrics>0], lons[metrics>0], boundary, factor=100, rodal_len=10)
    save_raster(folder + file[:-4] + '_PD.tif', p_raster, geotrans, data_type = gdal.GDT_UInt32)

    alpha_vol, alpha_dap, alpha_ab = alpha_value(dictionary)

    vol = alpha_vol*(h_max**2.5118603068)*(dictionary['EDAD_2020']**0.2808104916)*(d_rod**(-0.1792966771))
    dap = alpha_dap / (1 + np.exp(1.4204594210 - (0.0731514616*h_max) - (-0.0002719001*d_rod)))
    ab = alpha_ab*(h_max**1.494674941)*(d_rod**(-0.244709997))

    # Add the metrics to DataFrame
    df_table = pd.DataFrame()
    df_table['id'] = index.tolist()
    df_table['Y'] = lats.tolist()
    df_table['X'] = lons.tolist()
    df_table['h_max'] = h_max.tolist()
    df_table['prun'] = metrics.tolist()
    df_table['vol'] = vol.tolist()
    df_table['ab'] = ab.tolist()
    df_table['dap'] = dap.tolist()
    df_table['cde'] = (h_max/dap).tolist()
    df_table['d_rod'] = d_rod.tolist()
    
    # Reorder columns and add some Stablishment keys
    predio, seccion, rodal = file[:5], file[6:8], file[9:13]
    cols = df_table.columns.tolist()
    df_table['PREDIO'] = predio
    df_table['SECCION'] = seccion
    df_table['RODAL'] = rodal
    ncols = ['PREDIO', 'SECCION', 'RODAL'] + cols
    df_table = df_table[ncols]

    # Add metadata to Dataframe
    df_table.n_trees = n_trees
    df_table.area = area
    df_table.tree_avg = tree_avg
    df_table.prun_avg = prun_avg
    df_table.n_prun = n_prun
    df_table.h_avg = np.around(np.mean(h_max),1)
    df_table.vol_avg = np.around(vol/area)

    # Save files
    df_table.to_csv(folder + file[:-4] + '.csv', sep=',',index=False)

    return df_table

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Airborne LiDAR Sensor Forestry Processing Algorithm')
    parser.add_argument('input_path', type=str, help="Input dir for laz files")
    parser.add_argument('output_path', type=str, help="Output dir to save the results")
    parser.add_argument('--dist_min', type=float, default = 1.20, help="Minimum distance between individual trees")
    parser.add_argument('--perc_factor', type=float, default = 25, help="Percentile tolerance")
    parser.add_argument('--h_min', type=float, default = 10, help="Minimum height for a tree")
    parser.add_argument('--scale', type=float, default = 0.15, help="Voxelization scale")

    args = parser.parse_args()

    # File paths
    input_folder = args.input_path
    output_folder = args.output_path

    # Check the syntax of the paths
    output_folder, output_folder = check_path(input_folder), check_path(output_folder)

    # Create folder in case it does not exist and list files
    create_folder(output_folder)
    files = files_list(input_folder)

    # Load forest table
    file_path = 'ROD_KEY.csv'
    df_forest = pd.read_csv(file_path, sep = ',')

    # Create and fill the density and average summary table
    i_rodal = 0
    df_summary = pd.DataFrame(columns=['PREDIO', 'SECCION', 'RODAL', 'N_TREES', 'N_POD', 'AREA', 'DENSIDAD_PROMEDIO', 'DENSIDAD_POD', 'H_PROMEDIO', 'VOLUMEN_HA' ])

    frames, corrupted = [], []

    for file in files:
        
        try:
            key = file[0:5] + file[6:8] + file[9:13]
            dictionary = df_forest[df_forest['CLAVE'] == int(key)].to_dict('records')[0]
            df = run(input_folder, file, args.scale, args.dist_min, args.perc_factor, args.h_min, dictionary, output_folder)
            df_summary.loc[i_rodal] = [file[0:5], file[6:8], file[9:13], df.n_trees, df.n_prun, df.area, df.tree_avg, df.prun_avg, df.h_avg, df.vol_avg]
            i_rodal = i_rodal + 1
            frames.append(df)
            print(file, ' has been succesfully processed.')
            df_summary.to_csv(output_folder + 'DENSIDAD_RESUMEN.csv', sep=',',index=False)

        except:
            print(file, ' was corrupted, it has been skipped.')
            corrupted.append(file)

    if len(corrupted) > 0:

        write_txt(output_folder + 'ARCHIVOS_CORRUPTOS.txt', corrupted)
        print('The following files were not processed:')
        for corrupt in corrupted:
            print(file)

    result = pd.concat(frames)

    # Save summary dataframe
    df_summary.to_csv(output_folder + 'DENSIDAD_RESUMEN.csv', sep=',',index=False)

    # Join all the results into a single Dataframe
    result = pd.concat(frames)
    result.to_csv(output_folder + 'CONSOLIDADO.csv', sep=',', index=False)

    # Save individual tree  Point shapefile
    gdf = gpd.GeoDataFrame(result, geometry=gpd.points_from_xy(result.X, result.Y))
    gdf.to_file(output_folder + 'CONSOLIDADO.shp')