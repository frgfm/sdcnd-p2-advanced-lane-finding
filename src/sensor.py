#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''
Sensor objects
'''

from pathlib import Path
import numpy as np
import cv2


class Camera(object):
    """Implements a camera object"""
    def __init__(self):
        self.camera_matrix = None
        self.distortion_coeff = None

    def calibrate_on(self, img_folder, grid_shape):
        """Calibrates camera using chessboard images

        Args:
            img_folder (pathlib.Path): path to the chessboard images' folder
            grid_shape (tuple<int>): chessboard grid dimension
        """

        # Verify input argument
        if not isinstance(img_folder, Path):
            raise TypeError(f"expected type pathlib.Path, received {type(img_folder)}")
        elif not img_folder.is_dir():
            raise FileNotFoundError(f"unable to access folder {img_folder}")

        obj_points = []
        chess_corners = []

        # Each image has the same chessboard grid layout
        _points = np.zeros((grid_shape[0] * grid_shape[1], 3), dtype=np.float32)
        _points[:, :2] = np.mgrid[0: grid_shape[0], 0: grid_shape[1]].T.reshape(-1, 2)

        # Loop on all images
        for file in img_folder.glob('**/*'):
            # Locate chessboard corners
            img = cv2.imread(file.as_posix())
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            ret, corners = cv2.findChessboardCorners(gray, grid_shape, None)
            # If there is a match, store the distorted coordinates
            if ret:
                img = cv2.drawChessboardCorners(img, grid_shape, corners, ret)
                chess_corners.append(corners)
                obj_points.append(_points)
        # Use stored points for calibration
        _, self.camera_matrix, self.distortion_coeff, _, _ = cv2.calibrateCamera(obj_points, chess_corners,
                                                                                 gray.shape[::-1], None, None)

    def __call__(self, img):
        """Applies camera calibration to an image

        Args:
            img (numpy.ndarray[H, W, C]): input image

        Returns:
            numpy.ndarray[H, W, C]: undistorted image
        """

        # Check for calibration
        if self.camera_matrix is None or self.distortion_coeff is None:
            raise AssertionError("The camera has not been calibrated yet.")
        else:
            # Applies inversed distortion matrix
            return cv2.undistort(img, self.camera_matrix, self.distortion_coeff, None, self.camera_matrix)
