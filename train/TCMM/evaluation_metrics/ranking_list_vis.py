import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import cv2
import os

GRID_SPACING = 10
QUERY_EXTRA_SPACING = 90
BW = 5
MARGIN = 20
TEXT_HEIGHT = 30
GREEN = (0, 255, 0)
RED = (0, 0, 255)
width = 128
height = 256
topk = 5
query_gallery_vis_dir = "ranking_list_vis"

def mkdir_if_missing(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

def top5_plot(i, top5_indices=None, top5_matches=None, query_img_paths=None, gallery_img_paths=None):
    qimg = cv2.imread(query_img_paths[i])
    qimg = cv2.resize(qimg, (width, height))
    qimg = cv2.copyMakeBorder(qimg, BW, BW, BW, BW, cv2.BORDER_CONSTANT, value=(0, 0, 0))
    qimg = cv2.resize(qimg, (width, height))
    
    grid_img = 255 * np.ones(
        (height + MARGIN*2 + TEXT_HEIGHT,
            width + QUERY_EXTRA_SPACING + topk * width + (topk - 1) * GRID_SPACING + MARGIN*2, 3), dtype=np.uint8)
    
    grid_img[MARGIN + TEXT_HEIGHT : MARGIN + TEXT_HEIGHT + height, MARGIN : MARGIN + width, :] = qimg
    cv2.putText(grid_img, "Query Image", (MARGIN, MARGIN + TEXT_HEIGHT - 10), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

    for j, idx in enumerate(top5_indices):
        border_color = GREEN if top5_matches[j] else RED

        gimg = cv2.imread(gallery_img_paths[idx])
        gimg = cv2.resize(gimg, (width, height))
        gimg = cv2.copyMakeBorder(gimg, BW, BW, BW, BW, cv2.BORDER_CONSTANT, value=border_color)
        gimg = cv2.resize(gimg, (width, height))
        
        start_x = MARGIN + width + QUERY_EXTRA_SPACING + j * (width + GRID_SPACING)
        start_y = MARGIN + TEXT_HEIGHT

        grid_img[start_y:start_y + height, start_x:start_x + width, :] = gimg
        cv2.putText(grid_img, f"Gallery Rank {j + 1}", (start_x, MARGIN + TEXT_HEIGHT - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

    mkdir_if_missing(query_gallery_vis_dir)
    output_filename = os.path.splitext(os.path.basename(query_img_paths[i]))[0] + ".png"
    output_path = os.path.join(query_gallery_vis_dir, output_filename)
    cv2.imwrite(output_path, grid_img)