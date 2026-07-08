import cv2
import numpy as np
import skimage as ski
import matplotlib.pyplot as plt
from skimage.morphology import white_tophat, disk
from skimage.feature import blob_dog, blob_log, blob_doh
from candidates import Candidate
from tqdm import tqdm
from Tile import Tile
from PIL import Image
import os
import pandas as pd

class PNNPipeline:
    def __init__(self, clip_limit = 5, gaussian_sigma = 1, rolling_ball_radius = 50,
                 hessian=True, laplacian=False, difference_of_gaussian=False, 
                 blob_thresh=0.07, blob_min_sigma = 4, blob_max_sigma = 12, num_sigma = 10,
                 roi_size = 96):
        
        self.clip_limit = clip_limit
        self.gaussian_sigma = gaussian_sigma
        self.radius = rolling_ball_radius
        self.image = None
        self.doh = hessian
        self.log = laplacian
        self.dog = difference_of_gaussian
        self.blob_thresh = blob_thresh
        self.blob_min_sigma = blob_min_sigma
        self.blob_max_sigma = blob_max_sigma
        self.num_sigma = num_sigma
        self.roi_size = roi_size

    def load_image(self, tile: Tile):
        self.image = tile.image
        self.tile = tile
        self.original_image = tile.original_image
        self.padded_image = np.pad(self.original_image, (int(self.roi_size/2), int(self.roi_size/2)))
        self.enhanced_image = None
        self.candidates = []
        
    def preprocess(self):
        print("Preprocessing...")
        print("Loading image...")
        if self.padded_image is None:
            raise ValueError("No image has been loaded")
        print("Image loaded successfully.")
        print("Histogram equalization...")
        clahe = cv2.createCLAHE(clipLimit = self.clip_limit)
        clahe_img = np.clip(clahe.apply(self.image))

        print("Histogram equalization done.")
        print("Applying gaussian...")
        gaussian_img = ski.filters.gaussian(clahe_img, self.gaussian_sigma)
        print("Applying gaussian done.")
        print("Preprocessing complete.")
        # str_el = disk(self.radius)
        # top_hat_img = white_tophat(gaussian_img, str_el)
        self.enhanced_image = gaussian_img

    def detect_candidates(self):
        print("Detecting candidates...")
        if self.doh:
            blobs = blob_doh(self.enhanced_image, 
                                 min_sigma=self.blob_min_sigma, 
                                 max_sigma=self.blob_max_sigma, 
                                 threshold=self.blob_thresh)
        elif self.log == True:
            blobs = blob_log(self.enhanced_image, 
                                 min_sigma=self.blob_min_sigma, 
                                 max_sigma=self.blob_max_sigma, 
                                 threshold=self.blob_thresh,
                                 num_sigma=self.num_sigma)
        elif self.dog == True:
            blobs = blob_dog(self.enhanced_image, 
                                 min_sigma=self.blob_min_sigma, 
                                 max_sigma=self.blob_max_sigma, 
                                 threshold=self.blob_thresh)                       
        self.candidates = []
        print("Detecting candidates done.")
        print("Adapting candidates...")
        for blob in blobs:
            candidate = Candidate(x = self.tile.x + int(blob[1]),
                                  y = self.tile.y + int(blob[0]),
                                  radius = blob[2],
                                  confidence = None,
                                  image_id = None)
            self.candidates.append(candidate)
        print("__________________________")
        print("Adapting candidates done.")    
        print(f"{len(self.candidates)} candidates found.")
                    
    def extract_rois(self):
        half_roi = int(self.roi_size/2)
        print("Extracting ROIs...")
        for idx, candidate in enumerate(self.candidates):
            roi = self.padded_image[candidate.y: candidate.y + self.roi_size,
                                    candidate.x: candidate.x + self.roi_size]
            candidate.roi = roi
            candidate.image_id = idx #TODO: Make a proper image ID based on original image
        
    def export_rois(self, path_folder, mouse_age, mouse_id, condition):
        # cols = ["Filename", "mouse", 'age', 'condition','x','y', 'cand_id']        
        # metadata = pd.DataFrame(columns=cols)
        rows_list = []
        for idx, candidate in enumerate(self.candidates):
            im = Image.fromarray(candidate.roi)
            filename = f"candidate{candidate.image_id}_mouse{mouse_id}_{mouse_age}_{condition}_x{candidate.x}_y{candidate.y}.tiff" 
            dict_row = {"Filename" : filename,
                        "mouse" : mouse_id,
                        "age" : mouse_age,
                        "condition" : condition,
                        "x" : candidate.x,
                        "y" : candidate.y,
                        "cand_id" : candidate.image_id} 
            abs_loc = os.path.join(path_folder, filename)
            rows_list.append(dict_row)
            im.save(abs_loc)
        metadata = pd.DataFrame(rows_list)
        metadata.to_csv(os.path.join(path_folder, "metadata.csv"), index=False)
    
    def segment(self):
        for candidate in self.candidates:
            candidate.mask = self.segment_candidate(candidate)

    # Adapter is numpy.ndarray to my object