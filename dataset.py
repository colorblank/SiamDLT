import torch
import torch.utils.data
import os
import csv
import numpy as np
from PIL import Image
import time

DAVIS_PATH = '/home/cly/datacenter/DAVIS'

IMG_MEAN = np.array((104.00698793,116.66876762,122.67891434), dtype=np.float32)

class DAVIS2017(torch.utils.data.Dataset):
    def __init__(self, debug=False):
        #get video list
        self.videos = []
        with open(os.path.join(DAVIS_PATH, 'ImageSets', '2017', 'train.txt'), 'r') as file:
            lines = csv.reader(file)
            for line in lines:
                self.videos.append(line[0])
        self.debug = debug

        np.random.seed(1337)

    def __len__(self):
        #return maximum of iterations
        return 100000

    # usage example:
    # (img1, anno1), (img2, anno2) = dataset.DAVIS2017().__getitem__(0)
    def __getitem__(self, index):
        #return a random frame pair from same video
        #img1 , anno1, img2, anno2
        random_video = self.videos[np.random.randint(0, len(self.videos))]
        num_frames = len(os.listdir(os.path.join(DAVIS_PATH, 'JPEGImages', '480p', random_video)))

        #debug: only training boat
        if self.debug:
            random_video = 'boat'
            num_frames = 75

        i = np.random.randint(0, num_frames)
        j = np.random.randint(0, num_frames)
        return self.get_img_anno_pair(random_video, i), self.get_img_anno_pair(random_video, j), random_video

    def get_img_anno_pair(self, video, idx):
        return self.get_image(video, idx), self.get_anno(video, idx)

    def get_image(self, video, idx):
        img_file = os.path.join(DAVIS_PATH, 'JPEGImages', '480p', video, '{:05}.jpg'.format(idx))
        image = np.asarray(Image.open(img_file).convert('RGB'), np.float32)
        image -= IMG_MEAN
        image = image.transpose((2, 0, 1))
        return torch.from_numpy(image)

    def get_anno(self, video, idx):
        anno_file = os.path.join(DAVIS_PATH, 'Annotations', '480p', video, '{:05}.png'.format(idx))
        anno = np.asarray(Image.open(anno_file).convert('P'), np.float32)
        return torch.from_numpy(anno)

    def get_original_img(self, video, idx):
        img_file = os.path.join(DAVIS_PATH, 'JPEGImages', '480p', video, '{:05}.jpg'.format(idx))
        image = np.asarray(Image.open(img_file).convert('RGB'), np.int)
        return image

