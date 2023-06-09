import os.path as osp
from glob import glob

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


class DataLoader(Dataset):
    def __init__(self, data_path, gt_pose=False,
                 scale_factor=0, crop=0, depth_scale=1000.0, max_depth=10, **kwargs) -> None:
        self.crop = crop
        self.depth_scale = depth_scale
        self.data_path = data_path
        self.scale_factor = scale_factor
        self.gt_pose = gt_pose
        num_imgs = len(glob(osp.join(data_path, 'rgb/*.jpg')))
        self.max_depth = max_depth
        self.K = self.load_intrinsic()
        self.depth_files = sorted(glob(f'{data_path}/depth/frame-*.depth.pgm'))
        self.image_files = sorted(glob(f'{data_path}/rgb/frame-*.color.jpg'))
        self.pose_file = osp.join(data_path, "gt.txt")
        self.num_imgs = num_imgs
        
        self.load_poses(self.pose_file)

    def load_poses(self, path):
        poses = []
        with open(path, "r") as f:
            lines = f.readlines()

        for line in lines:
            c2w = np.array(list(map(float, line.split()))).reshape(3, 4)
            c2w = np.vstack((c2w, np.array([0,0,0,1])))
            c2w = torch.from_numpy(c2w).float()
            poses.append(c2w)

        poses = torch.stack(poses, dim=0).reshape(len(poses),16)
        self.poses = np.array(poses)

    def load_intrinsic(self):
        self.K = np.eye(3)
        self.K[0, 0] = 577.590698
        self.K[1, 1] = 578.729797
        self.K[0, 2] = 318.905426
        self.K[1, 2] = 242.683609

        if self.crop > 0:
            self.K[0, 2] = self.K[0, 2] - self.crop
            self.K[1, 2] = self.K[1, 2] - self.crop
        return self.K

    def load_depth(self, index):
        depth = cv2.imread(self.depth_files[index], -1) / self.depth_scale
        depth[depth > self.max_depth] = 0
        if self.scale_factor > 0:
            skip = 2**self.scale_factor
            depth = depth[::skip, ::skip]
        if self.crop > 0:
            depth = depth[self.crop:-self.crop, self.crop:-self.crop]
        return depth

    def get_init_pose(self):
        return self.poses[0]

    def load_image(self, index):
        img = cv2.imread(self.image_files[index], -1)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (640, 480), cv2.INTER_AREA)  # / 255.0
        if self.scale_factor > 0:
            factor = 2**self.scale_factor
            size = (640 // factor, 480 // factor)
            img = cv2.resize(img, size, cv2.INTER_AREA)
        if self.crop > 0:
            img = img[self.crop:-self.crop, self.crop:-self.crop]
        return img / 255.0

    def __len__(self):
        return len(self.depth_files)

    def __getitem__(self, index):
        img = torch.from_numpy(self.load_image(index)).float()
        depth = torch.from_numpy(self.load_depth(index)).float()
        pose = self.poses[index].reshape(4,4)
        # pose = self.poses[index] if self.gt_pose else None
        return index, img, depth, self.K, pose


if __name__ == '__main__':
    import sys
    loader = DataLoader(sys.argv[1], 1)
    for data in loader:
        index, img, depth = data
        print(index, img.shape)
        cv2.imshow('img', img)
        cv2.waitKey(1)
