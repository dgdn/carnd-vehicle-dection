import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import pickle
import glob
import cv2
from utils import *
from scipy.ndimage.measurements import label
from moviepy.editor import VideoFileClip

def add_heat(heatmap, bbox_list):
    # Iterate through list of bboxes
    for box in bbox_list:
        # Add += 1 for all pixels inside each bbox
        # Assuming each "box" takes the form ((x1, y1), (x2, y2))
        heatmap[box[0][1]:box[1][1], box[0][0]:box[1][0]] += 1

    # Return updated heatmap
    return heatmap# Iterate through list of bboxes
    
def apply_threshold(heatmap, threshold):
    # Zero out pixels below the threshold
    heatmap[heatmap <= threshold] = 0
    # Return thresholded map
    return heatmap

def draw_labeled_bboxes(img, labels):
    # Iterate through all detected cars
    for car_number in range(1, labels[1]+1):
        # Find pixels with each car_number label value
        nonzero = (labels[0] == car_number).nonzero()
        # Identify x and y values of those pixels
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        # Define a bounding box based on min/max x and y
        bbox = ((np.min(nonzerox), np.min(nonzeroy)), (np.max(nonzerox), np.max(nonzeroy)))
        # Draw the box on the image
        cv2.rectangle(img, bbox[0], bbox[1], (0,0,255), 6)
    # Return the image
    return img

# Define a single function that can extract features using hog sub-sampling and make predictions
def find_cars_bboxes(img, ystart, ystop, scale, svc, X_scaler, orient, pix_per_cell, cell_per_block, spatial_size, hist_bins):
    
    # Scale to 0 between 1 to keep the same as training data
    img = img.astype(np.float32)/255
    
    img_tosearch = img[ystart:ystop,:,:]
    ctrans_tosearch = convert_color(img_tosearch, color_space='YCrCb')
    if scale != 1:
        imshape = ctrans_tosearch.shape
        ctrans_tosearch = cv2.resize(ctrans_tosearch, (np.int(imshape[1]/scale), np.int(imshape[0]/scale)))
        
    ch1 = ctrans_tosearch[:,:,0]
    ch2 = ctrans_tosearch[:,:,1]
    ch3 = ctrans_tosearch[:,:,2]

    # Define blocks and steps as above
    nxblocks = (ch1.shape[1] // pix_per_cell) - cell_per_block + 1
    nyblocks = (ch1.shape[0] // pix_per_cell) - cell_per_block + 1 
    nfeat_per_block = orient*cell_per_block**2
    
    # 64 was the orginal sampling rate, with 8 cells and 8 pix per cell
    window = 64
    nblocks_per_window = (window // pix_per_cell) - cell_per_block + 1
    cells_per_step = 2  # Instead of overlap, define how many cells to step
    nxsteps = (nxblocks - nblocks_per_window) // cells_per_step + 1
    nysteps = (nyblocks - nblocks_per_window) // cells_per_step + 1
    
    # Compute individual channel HOG features for the entire image
    hog1 = get_hog_features(ch1, orient, pix_per_cell, cell_per_block, feature_vec=False)
    hog2 = get_hog_features(ch2, orient, pix_per_cell, cell_per_block, feature_vec=False)
    hog3 = get_hog_features(ch3, orient, pix_per_cell, cell_per_block, feature_vec=False)

    box_list = []
    for xb in range(nxsteps):
        for yb in range(nysteps):
            ypos = yb*cells_per_step
            xpos = xb*cells_per_step
            # Extract HOG for this patch
            hog_feat1 = hog1[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel() 
            hog_feat2 = hog2[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel() 
            hog_feat3 = hog3[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel() 
            hog_features = np.hstack((hog_feat1, hog_feat2, hog_feat3))

            xleft = xpos*pix_per_cell
            ytop = ypos*pix_per_cell

            # Extract the image patch
            subimg = cv2.resize(ctrans_tosearch[ytop:ytop+window, xleft:xleft+window], (64,64))
          
            # Get color features
            spatial_features = bin_spatial(subimg, size=spatial_size)
            hist_features = color_hist(subimg, nbins=hist_bins)

            # Scale features and make a prediction
            test_features = X_scaler.transform(np.hstack((spatial_features, hist_features, hog_features)).reshape(1, -1))    
            test_prediction = svc.predict(test_features)
            scores = svc.decision_function(test_features)
            if test_prediction == 1 and scores[0] > score_threshold :
                xbox_left = np.int(xleft*scale)
                ytop_draw = np.int(ytop*scale)
                win_draw = np.int(window*scale)
                box_list.append(((xbox_left, ytop_draw+ystart),(xbox_left+win_draw,ytop_draw+win_draw+ystart)))

    return box_list

# load a pe-trained svc model from a serialized (pickle) file
dist_pickle = pickle.load(open("svc_pickle.p", "rb" ))

# get attributes of our svc object
svc = dist_pickle["svc"]
X_scaler = dist_pickle["scaler"]
orient = dist_pickle["orient"]
pix_per_cell = dist_pickle["pix_per_cell"]
cell_per_block = dist_pickle["cell_per_block"]
spatial_size = dist_pickle["spatial_size"]
hist_bins = dist_pickle["hist_bins"]

ystart = 400
ystop = 656
scales = [1, 1.5]
heat_threshold = 2
score_threshold = 0.5

def find_multi_scale_bbxes():

    fig = plt.figure(figsize=(10,10))
    images = glob.glob('test_images/*.jpg')
    length = len(images)
    for i, file in enumerate(images):
        img = mpimg.imread(file)
        box_list = []
        for scale in scales:
            boxes = find_cars_bboxes(img, ystart, ystop, scale, svc, X_scaler, orient, pix_per_cell, cell_per_block, spatial_size, hist_bins)
            box_list += boxes
        heat = np.zeros_like(img[:,:,0]).astype(np.float)
        # Add heat to each box in box list
        heat = add_heat(heat,box_list)
        # Apply threshold to help remove false positives
        heat = apply_threshold(heat, heat_threshold)


        heatmap_img = cv2.applyColorMap(heat/np.max(heat)*255, cv2.COLORMAP_JET)
        heatmap_shape = (int(heatmap_img.shape[0]/3), int(heatmap_img.shape[1]/3))
        heatmap_img = cv2.resize(heatmap_img, heatmap_shape)

        # Visualize the heatmap when displaying    

        heatmap = np.clip(heat, 0, 255)
        # Find final boxes from heatmap using label function
        labels = label(heatmap)
        label_bbx_img = draw_labeled_bboxes(img, labels)
        label_bbx_origin_img = draw_boxes(img,box_list)

        label_bbx_img_test = np.copy(label_bbx_img)

        label_bbx_img_test[0:heatmap_shape[1], 0:heatmap_shape[0], :] = heatmap_img
        fig.add_subplot(length, 2, i*2+1)
        plt.imshow(label_bbx_origin_img)
        fig.add_subplot(length, 2, i*2+2)
        plt.imshow(label_bbx_img)

    plt.show()

heats = []
heat_avg_count = 20
def get_average_heat(heat):
    global heats
    heats.append(np.copy(heat))
    if len(heats) > heat_avg_count:
        heats = heats[-heat_avg_count:]
    return np.average(heats, axis=0) 

def heatmap_img(heat):
    return cv2.applyColorMap(heat/np.max(heat)*255, cv2.COLORMAP_JET)

def pipeline(img):

    box_list = []
    for scale in scales:
        boxes = find_cars_bboxes(img, ystart, ystop, scale, svc, X_scaler, orient, pix_per_cell, cell_per_block, spatial_size, hist_bins)
        box_list += boxes

    heat = np.zeros_like(img[:,:,0]).astype(np.float)
    # Add heat to each box in box list
    heat = add_heat(heat,box_list)

    heatmap_img = cv2.applyColorMap(heat/np.max(heat)*255, cv2.COLORMAP_JET)
    heatmap_shape = (heatmap_img.shape[0]/3, heatmap_img.shape[1]/3)
    heatmap_img = cv2.resize(heatmap_img, heatmap_shape)

    #heat = get_average_heat(heat)

    # Apply threshold to help remove false positives
    heat = apply_threshold(heat, heat_threshold)
    # Visualize the heatmap when displaying    
    heat = np.clip(heat, 0, 255)
    # Find final boxes from heatmap using label function
    labels = label(heat)
    label_bbx_img = draw_labeled_bboxes(img, labels)

    label_bbx_img[0:heatmap_shape[0], 0:heatmap_shape[1], :] = heatmap_img
    return label_bbx_img

def main():

    # Process the video stream using the provided pipline
    easy_output = 'project_video_output.mp4'
    clip1 = VideoFileClip('test_video.mp4')
    easy_clip = clip1.fl_image(pipeline)
    easy_clip.write_videofile(easy_output, audio=False)

if __name__ == "__main__":
    find_multi_scale_bbxes()
    #main()
    #test()
