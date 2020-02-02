#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''
Utilities
'''

import numpy as np


def predict_from_fit(fit, samples):
    """Uses a polynomial regression coefficients to compute predictions

    Args:
        fit (numpy.ndarray[N]): (N-1)th order polynomial fit coefficients
        samples (numpy.ndarray[K]): input samples

    Returns:
        numpy.ndarray[K]: output predictions
    """

    return sum(fit[len(fit) - idx - 1] * samples ** idx if idx > 0 else fit[-1]
               for idx in range(len(fit)))


def curvature_from_fit(fit, y):
    """Compute curvature radius

    Args:
        fit (numpy.ndarray[3]<float>): polynomial regression fit coefficients
        y (float): point where curvature will be evaluated

    Returns:
        float: curvature radius
    """

    if len(fit) != 3:
        raise AssertionError(f"expected fit coefficients to be of shape (3,), but received {fit.shape}")

    f_prime = 2 * fit[0] * y + fit[1]
    f_second = 2 * fit[0]
    return (1 + (f_prime ** 2)) ** 1.5 / abs(f_second)


def get_depth_vertices(img_shape, lat_margin=0.08, horizon=(0.55, 0.5), vert_range=(0.62, 0.9)):
    """Compute depth view vertices

    Args:
        img_shape (tuple<int>): shape of the input image
        lat_margin (float, optional): relative lateral offset of the bottom of the mask
        horizon (tuple<float>, optional): relative coordinates of apparent horizon
        vert_range (tuple<float>, optional): relative range of vertical masking

    Returns:
        numpy.ndarray: vertices of depth view mask
    """

    # Compute cut coordinates
    leftcut_min = lat_margin + (1 - vert_range[0]) / (1 - horizon[0]) * (horizon[1] - lat_margin)
    leftcut_max = lat_margin + (1 - vert_range[1]) / (1 - horizon[0]) * (horizon[1] - lat_margin)

    vertices = np.array([[
        (leftcut_max * img_shape[1], vert_range[1] * img_shape[0]),
        (leftcut_min * img_shape[1], vert_range[0] * img_shape[0]),
        ((1 - leftcut_min) * img_shape[1], vert_range[0] * img_shape[0]),
        ((1 - leftcut_max) * img_shape[1], vert_range[1] * img_shape[0])]], dtype=np.float32)

    return vertices
