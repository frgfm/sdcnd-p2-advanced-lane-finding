# Advanced lane finding
[![License](https://img.shields.io/badge/License-MIT-brightgreen.svg)](LICENSE) [![Codacy Badge](https://api.codacy.com/project/badge/Grade/3bc1cc0064b247d3b24ee58d716d5f34)](https://www.codacy.com/manual/frgfm/sdcnd-p2-advanced-lane-finding?utm_source=github.com&amp;utm_medium=referral&amp;utm_content=frgfm/sdcnd-p2-advanced-lane-finding&amp;utm_campaign=Badge_Grade) [![CircleCI](https://circleci.com/gh/frgfm/sdcnd-p2-advanced-lane-finding.svg?style=shield)](https://circleci.com/gh/frgfm/sdcnd-p2-advanced-lane-finding)

This repository is an implementation of non-learning pipeline for the advanced lane finding project of Udacity Self-Driving Car Nanodegree (cf. [repo](<https://github.com/udacity/CarND-Advanced-Lane-Lines>)).

![video-sample](static/images/video-sample.gif)



## Table of Contents

- [Getting Started](#getting-started)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
- [Usage](#usage)
- [Approach](#Approach)
- [Credits](#credits)
- [License](#license)



## Getting started

### Prerequisites

- Python 3.6 (or more recent)
- [pip](https://pip.pypa.io/en/stable/)

### Installation

You can install the project requirements as follows:

```shell
git clone https://github.com/frgfm/sdcnd-p2-advanced-lane-finding.git
cd sdcnd-p2-advanced-lane-finding
pip install -r requirements.txt
```



## Usage

All script arguments can be found using the `--help` flag:

```shell
python main.py --help
```

Below you can find an example to detect lanes in an image and in a video:

```shell
python main.py test_images/test2.jpg
python main.py test_videos/project_video.mp4 --video
```



## Approach

This detection approach was designed to avoid a learning procedure and only requires camera calibration to perform lane segmentation.



![](https://video.udacity-data.com/topher/2016/December/5840ae19_screen-shot-2016-12-01-at-3.10.19-pm/screen-shot-2016-12-01-at-3.10.19-pm.png)



![](https://video.udacity-data.com/topher/2018/June/5b2343e8_screen-shot-2017-01-28-at-11.49.20-am/screen-shot-2017-01-28-at-11.49.20-am.png)



Radius of curvature 
$$
R_{curve} = \frac{[1 + (\frac{dx}{dy})^2]^{3/2}}{|\frac{d^2x}{dy^2}|}
$$
But since $f(y) = Ay^2 + By + C$, then $f'(y) = 2Ay + B$ and $f''(y) = 2A$

thus
$$
R_{curve} = \frac{[1 + (2Ay + B)^2]^{3/2}}{|2A|}
$$
Using the regression results, we can get the curvature with the above formula



### Result

![img-sample](static/images/img-sample.jpg)



## Credits

This implementation is vastly based on the following methods:

- [Camera Calibration](https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_calib3d/py_calibration/py_calibration.html)

- [Colorspaces](https://en.wikipedia.org/wiki/HSL_and_HSV)
- [Canny edge detection](https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_imgproc/py_canny/py_canny.html)
- [Polynomial regression](https://en.wikipedia.org/wiki/Polynomial_regression)



## License

Distributed under the MIT License. See `LICENSE` for more information.