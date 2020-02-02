#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''
Detector object
'''

import numpy as np
import cv2

from sensor import Camera
from lane import Line, Lane
from utils import get_depth_vertices


class LaneDetector(object):
    """Implements a lane detector

    Args:
        sobel_thresh (tuple<int>, optional): lower and upper threshold for edge detection with sobel
        saturation_thresh (tuple<int>, optional): lower and upper threshold for edge detection with channel values
        proj_xmargin (float, optional):
        proj_ymargin (float, optional):
        persp_latmargin (float, optional):
        persp_horizon (tuple<float>, optional):
        persp_vertrange (tuple<float>, optional):
        num_windows (int, optional): number of windows used for polynomial regression
        margin (int, optional): margin used around line center for polynomial regression
        minpix (int, optional): minimum number of line points to attempt regression
    """

    def __init__(self, sobel_thresh=(20, 100), saturation_thresh=(170, 255),
                 proj_xmargin=0.2, proj_ymargin=0.,
                 persp_latmargin=0.15, persp_horizon=(0.58, 0.5), persp_vertrange=(0.62, 1.),
                 num_windows=10, margin=100, minpix=50):

        # Edge detection
        self.sobel_thresh = sobel_thresh
        self.saturation_thresh = saturation_thresh

        # Perspective
        self.proj_xmargin = proj_xmargin
        self.proj_ymargin = proj_ymargin
        self.persp_latmargin = persp_latmargin
        self.persp_horizon = persp_horizon
        self.persp_vertrange = persp_vertrange

        # Lane fitting
        self.num_windows = num_windows
        self.margin = margin
        self.minpix = minpix
        self.reg_boxes = []

    def detect_edges(self, img):
        """Detect edges in picture

        Args:
            img (numpy.ndarray[H, W, C]): input image

        Returns:
            numpy.ndarray[H, W]: binary edge mask
        """
        img = np.copy(img)
        # Convert to HLS color space and separate the V channel
        hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
        l_channel = hls[:, :, 1]
        s_channel = hls[:, :, 2]
        # Sobel x
        sobelx = cv2.Sobel(l_channel, cv2.CV_64F, 1, 0)  # Take the derivative in x
        abs_sobelx = np.absolute(sobelx)  # Absolute x derivative to accentuate lines away from horizontal
        scaled_sobel = np.uint8(255 * abs_sobelx / np.max(abs_sobelx))

        # Threshold x gradient
        sxbinary = np.zeros_like(scaled_sobel)
        sxbinary[(scaled_sobel >= self.sobel_thresh[0]) & (scaled_sobel <= self.sobel_thresh[1])] = 1

        # Threshold color channel
        s_binary = np.zeros_like(s_channel)
        s_binary[(s_channel >= self.saturation_thresh[0]) & (s_channel <= self.saturation_thresh[1])] = 1

        # Combine HLS filtered mask with grayscale sobel mask
        combined_binary = np.zeros_like(sxbinary)
        combined_binary[(s_binary == 1) | (sxbinary == 1)] = 1

        return combined_binary

    def perspective_transform(self, img):
        """Applies perspective transform for a given part of the image

        Args:
            img (numpy.ndarray[H, W, C]): input image

        Returns:
            numpy.ndarray[H, W, C]: warped image
            numpy.ndarray[3, 3]: inversed warp matrix
        """

        # Quadrangle vertices coordinates in the source image
        source_coords = get_depth_vertices(img.shape, self.persp_latmargin, self.persp_horizon, self.persp_vertrange)

        # Quadrangle vertices coordinates in the destination image
        destination_coords = np.float32([
            [self.proj_xmargin * img.shape[1], (1 - self.proj_ymargin) * img.shape[0]],
            [self.proj_xmargin * img.shape[1], self.proj_ymargin * img.shape[0]],
            [(1 - self.proj_xmargin) * img.shape[1], self.proj_ymargin * img.shape[0]],
            [(1 - self.proj_xmargin) * img.shape[1], (1 - self.proj_ymargin) * img.shape[0]]
        ])
        # Given src and dst points we calculate the perspective transform matrix
        M = cv2.getPerspectiveTransform(source_coords, destination_coords)
        # Warp the image
        warped = cv2.warpPerspective(img, M, (img.shape[1], img.shape[0]))
        # We also calculate the oposite transform
        invwarp_m = cv2.getPerspectiveTransform(destination_coords, source_coords)
        # Return the resulting image and matrix
        return (warped, invwarp_m)

    @staticmethod
    def get_window_indices(indices, y_thresh, x, margin):
        """Locate non-zero indices meeting threshold conditions

        Args:
            indices (numpy.ndarray[2, K]): non-zero indices
            y_thresh (tuple<int>): lower and upper vertical indices
            x (int): expected horizontal middle of the line
            margin (int): horizontal margin around x

        Returns:
            numpy.ndarray(numpy.ndarray[2, M])
        """

        return ((indices[0] >= y_thresh[0]) & (indices[0] < y_thresh[1]) &
                (indices[1] >= x - margin) &
                (indices[1] < x + margin)).nonzero()[0]

    def fit_lane(self, bin_mask):
        """Fits a lane on a binary image

        Args:
            bin_mask (numpy.ndarray[H, W]): edge mask

        Returns:
            Lane: two-line lane
        """

        # Compute lower half histogram
        histogram = np.sum(bin_mask[bin_mask.shape[0] // 2:, :], axis=0)
        nonzero = bin_mask.nonzero()

        left_lane_inds = []
        right_lane_inds = []

        win_height = bin_mask.shape[0] // self.num_windows

        # Window initialization
        leftx_current = np.argmax(histogram[:bin_mask.shape[1] // 2])
        rightx_current = (bin_mask.shape[1] // 2) + np.argmax(histogram[bin_mask.shape[1] // 2:])

        # Store point values in each window
        for idx in range(self.num_windows):

            # Sliding window boundaries
            y_low = bin_mask.shape[0] - (idx + 1) * win_height
            y_high = bin_mask.shape[0] - idx * win_height

            # Add the nonzero indices
            left_lane_inds.append(self.get_window_indices(nonzero, (y_low, y_high), leftx_current, self.margin))
            self.reg_boxes.append(((leftx_current - self.margin, y_low), (leftx_current + self.margin, y_high)))
            right_lane_inds.append(self.get_window_indices(nonzero, (y_low, y_high), rightx_current, self.margin))
            self.reg_boxes.append(((rightx_current - self.margin, y_low), (rightx_current + self.margin, y_high)))

            # Update center
            if len(left_lane_inds[-1]) > self.minpix:
                leftx_current = nonzero[1][left_lane_inds[-1]].mean().astype(int)
            if len(right_lane_inds[-1]) > self.minpix:
                rightx_current = nonzero[1][right_lane_inds[-1]].mean().astype(int)

        # Fit each line
        left_line = Line(nonzero[1][np.concatenate(left_lane_inds)],
                         nonzero[0][np.concatenate(left_lane_inds)], bin_mask.shape[:2])
        right_line = Line(nonzero[1][np.concatenate(right_lane_inds)],
                          nonzero[0][np.concatenate(right_lane_inds)], bin_mask.shape[:2])

        # Return a lane object
        return Lane(left_line, right_line)

    @staticmethod
    def render(warped_lane, invwarp_mat, target_shape, radius=None, offset=None):
        """Renders final overlayed output

        Args:
            warped_lane (numpy.ndarray[M, N, 3]): birdeye view lane render
            invwarp_mat (numpy.ndarray[3, 3]): inversed perspective warping matrix
            target_shape (tuple<int>): target shape

        Returns:
            numpy.ndarray[H, W, 3]: overlayed output
        """

        # Warp the blank back to original image space using inverse perspective matrix (Minv)
        ui = cv2.warpPerspective(warped_lane, invwarp_mat, target_shape)

        # Dashboard
        cv2.rectangle(ui, (0, 0), (target_shape[0], 200), (50, 50, 50), cv2.FILLED)

        # Extras
        extra_str = []
        if isinstance(radius, float):
            extra_str.append(f"Curvature radius: {radius/1000:.2}km")
        if isinstance(offset, float):
            extra_str.append(f"Lateral shift: {offset:.2}m")

        top = 50
        for s in extra_str:
            cv2.putText(ui, s, (target_shape[1] - 100, top),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            top += 30

        return ui

    def __call__(self, img):
        """

        Args:
            img (numpy.ndarray[H, W, 3]): input image
            radius_aggregation (bool, optional): whether lane points should be aggregated for curvature radius

        Returns:
            numpy.ndarray[H, W, 3]: rendered output
        """

        # Get the binary edge mask
        bin_img = self.detect_edges(img)
        # Get the birdeye view transform, and inversed transformation matrix
        birdeye_img, invwarp_mat = self.perspective_transform(bin_img)

        # Fit a lane on the warped mask
        lane = self.fit_lane(birdeye_img)
        # Use the fit to determine the radius and shift from center
        radius = lane.get_curvature_radius()
        offset = lane.get_offset(img.shape[1])

        # Render the line-bounded color mask
        warped_lane = lane.render(birdeye_img.shape)

        # Warp the colored lane back to original image shape
        lane_mask = cv2.warpPerspective(warped_lane, invwarp_mat, (img.shape[1], img.shape[0]))
        result = cv2.addWeighted(img, 1, lane_mask, 0.3, 0)

        # Dark Dashboard
        board_height = 200
        result[:board_height, :] = result[:board_height, :] // 2

        # Canny edge detection render
        bird_img = 255 * np.repeat(birdeye_img[..., None], 3, axis=2)
        # Add regression boxes
        for top_left, bot_right in self.reg_boxes:
            cv2.rectangle(bird_img, top_left, bot_right, (255, 255, 0), 2)
        bird_img = cv2.resize(bird_img, (result.shape[1] // 3, board_height))
        result[:board_height, :result.shape[1] // 3] = cv2.addWeighted(result[:board_height, :result.shape[1] // 3],
                                                                       0.3, bird_img, 0.7, 0)

        # Birdeye view lane
        tmp_img = cv2.resize(cv2.addWeighted(self.perspective_transform(img)[0], 1, warped_lane, 0.3, 0),
                             (result.shape[1] // 3, board_height))
        result[:board_height, result.shape[1] // 3: 2 * result.shape[1] // 3 - 1] = cv2.addWeighted(
            result[:board_height, result.shape[1] // 3: 2 * result.shape[1] // 3 - 1], 0.3,
            tmp_img, 0.7, 0)

        # Radius & Offset
        extra_str = []
        if isinstance(radius, float):
            extra_str.append(f"Curvature radius: {radius/1000:.2}km")
        if isinstance(offset, float):
            extra_str.append(f"Lateral shift: {offset:.2}m")

        top = 50
        for s in extra_str:
            cv2.putText(result, s, (2 * img.shape[1] // 3 + 50, top),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            top += 50

        return result
