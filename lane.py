#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''
Lane objects
'''

from collections import deque
import numpy as np
import cv2
from utils import predict_from_fit, curvature_from_fit


class Line(object):
    """Implements a lane-separating line

    Args:
        x (numpy.ndarray[K]<float>): horizontal coordinates of points
        y (numpy.ndarray[K]<float>): vertical coordinates of points
        img_shape (tuple<int>): image shape
        memory (int, optional): number of regression coefficients kept for aggregation
    """

    def __init__(self, x, y, img_shape, memory=5):
        self.img_shape = img_shape
        self.coefs = deque(maxlen=memory)
        self.fit(x, y)
        self.curvature_y = 3.0
        self.samples = None

    def fit(self, x, y):
        """Fits a polynomial regression

        Args:
            x (numpy.ndarray[K]<float>): horizontal coordinates of points
            y (numpy.ndarray[K]<float>): vertical coordinates of points

        Returns:
            numpy.ndarray[3]<float>: regression coefficients
        """
        # TODO: safety checks
        self.coefs.append(np.polyfit(y, x, 2))

    @property
    def coef(self):
        return np.asarray(self.coefs).mean(axis=0)

    def get_samples(self, num_samples=None):
        """Used polynomial fit to render samples

        Args:
            num_samples (int, optional): number of samples to render

        Returns:
            numpy.ndarray[num_samples, 2]: rendered samples
        """

        # Default number of samples to vertical length
        if num_samples is None:
            num_samples = self.img_shape[0] - 1
        if self.samples is None or self.samples[0].shape[0] != num_samples:
            # Split vertical indices into samples
            y_samples = np.linspace(0, self.img_shape[0] - 1, num_samples).astype(float)
            # Average last regression coefficients
            # Store samples
            self.samples = np.stack((predict_from_fit(self.coef, y_samples),
                                     y_samples),
                                    axis=1)

        return self.samples


class Lane(object):
    """Implements a two-line lane

    Args:
        left_line (Line): fitted left line
        right_line (Line): fitted right line
        lane_width (float, optional): expected real-world lane width in meters
        mask_height (float, optional): expected real-world mask depth in meters
        min_radius (float, optional): minimum valid curvature radius
    """

    left_samples, right_samples = None, None

    def __init__(self, left_line, right_line, lane_width=3.7, mask_height=30., min_radius=200):
        self.left_line = left_line
        self.right_line = right_line
        self.lane_width = lane_width
        self.mask_height = mask_height
        self.min_radius = min_radius

    @staticmethod
    def _compute_curv_radius(points, y):
        """Compute curvature radius

        Args:
            points (numpy.ndarray[N, 2]<float>): points to use for curvature fit
            y (float): point where curvature will be evaluated

        Returns:
            float: curvature radius
        """
        # Fit in meter space
        fit = np.polyfit(points[:, 1], points[:, 0], 2)
        return curvature_from_fit(fit, y)

    def _get_samples(self, num_samples=100):
        """Generates point samples from both lines

        Args:
            num_samples (int, optional): number of samples to render

        Returns:
            numpy.ndarray[num_samples, 2]: left line samples
            numpy.ndarray[num_samples, 2]: right line samples
        """

        return self.left_line.get_samples(num_samples).copy(), self.right_line.get_samples(num_samples).copy()

    def get_curvature_radius(self, num_samples=100):
        """Compute the curvature radius of the lane

        Args:
            num_samples (int, optional): number of samples used for curvature computation
            radius_aggregation (bool, optional): should both lanes be aggregated for radius computation

        Returns:
            float: curvature radius of the lane
        """

        # Get some samples
        left_points, right_points = self._get_samples(num_samples)

        # For each vertical value, the difference between left and right Xs should be equal to lane_width
        xm_per_pix = self.lane_width / np.mean(right_points[:, 0] - left_points[:, 0])
        ym_per_pix = self.mask_height / (left_points[-1, 1] - left_points[0, 1])
        # Recycle pixel space regression coefficients and scale them to meter space
        left_coef, right_coef = self.left_line.coef, self.right_line.coef
        left_coef = [xm_per_pix / ym_per_pix ** 2 * left_coef[0],
                     xm_per_pix / ym_per_pix * left_coef[1],
                     xm_per_pix * left_coef[2]]
        right_coef = [xm_per_pix / ym_per_pix ** 2 * right_coef[0],
                      xm_per_pix / ym_per_pix * right_coef[1],
                      xm_per_pix * right_coef[2]]
        # Get the curvature without refitting
        left_radius = curvature_from_fit(left_coef, self.mask_height)
        right_radius = curvature_from_fit(right_coef, self.mask_height)
        radius = (left_radius + right_radius) / 2

        return radius

    def get_offset(self, img_width, num_samples=100):
        """Get lateral shift of the car from lane center

        Args:
            img_width (int): original image width
            num_samples (int, optional): number of samples used for computation

        Returns:
            float: lateral shift
        """

        left_points, right_points = self._get_samples(num_samples)
        # take the beginning of the lane
        lane_center = (right_points[-1, 0] + left_points[-1, 0]) / 2
        offset_pixels = (img_width / 2) - lane_center

        return offset_pixels * self.lane_width / np.mean(right_points[:, 0] - left_points[:, 0])

    def render(self, target_shape, num_samples=100):
        """Renders birdeye lane colored mask

        Args:
            target_shape (tuple<int>): birdeye image shape
            num_samples (int, optional): number of samples used for cv2.fillPoly

        Returns:
            numpy.ndarray[target_shape[0], target_shape[1], 3]: rendered colored mask
        """

        # Create an image to draw the lines on
        color_warp = np.zeros((*target_shape[:2], 3), dtype=np.uint8)

        left_points, right_points = self._get_samples(num_samples)
        # Recast the x and y points into usable format for cv2.fillPoly()
        # Draw the lane onto the warped blank image
        cv2.fillPoly(color_warp,
                     np.int_([np.hstack((left_points[None, ...],
                                         np.flipud(right_points)[None, ...]))]),
                     (0, 255, 0))

        return color_warp
