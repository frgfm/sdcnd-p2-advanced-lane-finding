#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''
Lane detection functions
'''

import matplotlib.image as mpimg
from functools import partial
from moviepy.editor import VideoFileClip
from detector import LaneDetector


def _process_frame(img, cam):
    """Compute the lane mask of an input image and overlay it on input image

    Args:
        img (numpy.ndarray[H, W, C]): input image
        colorspace (str, optional): colorspace to use for canny edge detection
        thickness (int, optional): thickness of lines on result image
        canny_low (int, optional): lower threshold for canny edge detection
        canny_high (int, optional): upper threshold for canny edge detection

    Returns:
        numpy.ndarray[H, W, 3]: lane mask
    """

    # Camera calibration
    cal_img = cam(img)
    # Detect lanes
    detector = LaneDetector()
    result = detector(cal_img)

    return result


def process_image(img_path, cam, thickness=3):
    """Read image and detect lanes on it

    Args:
        img_path (str): input image path
        thickness (int, optional): thickness of lines on result image

    Returns:
        numpy.ndarray[H, W, 3]: input image overlayed with result
    """

    # Load image and process
    img = mpimg.imread(img_path)

    return _process_frame(img, cam=cam)


def process_video(video_path, output_file, thickness=3, write_gif=False, cam=None):
    """Display lane detection results on input image

    Args:
        video_path (str): input video path
        output_file (str): output video path
        thickness (int, optional): thickness of lines on result image
        write_gif (bool, optional): should the output be a GIF rather than a video
    """

    # Read the video streams
    video = VideoFileClip(video_path)
    # Apply detection to each frame
    clip = video.fl_image(partial(_process_frame, cam=cam))
    # Output format
    if write_gif:
        file_split = output_file.split('.')
        if file_split[-1] != 'gif':
            output_file = '.'.join(file_split[:-1] + ['gif'])
        clip.write_gif(output_file, fps=5, program='ImageMagick')
    else:
        clip.write_videofile(output_file, audio=False)
