import unittest
import tempfile
import requests
from pathlib import Path
import matplotlib.image as mpimg
import numpy as np

from src.sensor import Camera


class SensorTester(unittest.TestCase):

    def test_camera(self):

        cam = Camera()
        self.assertRaises(TypeError, cam.calibrate_on, 'tmp_calibration_folder', (9, 6))
        self.assertRaises(FileNotFoundError, cam.calibrate_on, Path('tmp_calibration_folder'), (9, 6))

        with tempfile.TemporaryDirectory() as _folder:

            calibration_folder = Path(_folder)

            # Download calibration images
            URL = 'https://raw.githubusercontent.com/udacity/CarND-Advanced-Lane-Lines/master/camera_cal/'
            for file_name in ['calibration1.jpg', 'calibration2.jpg']:
                response = requests.get(f"{URL}{file_name}")
                with open(calibration_folder.joinpath(file_name), 'wb') as f:
                    f.write(response.content)

            # Check calibration
            cam.calibrate_on(calibration_folder, (9, 6))
            self.assertIsInstance(cam.camera_matrix, np.ndarray)
            self.assertEqual(cam.camera_matrix.shape, (3, 3))
            self.assertIsInstance(cam.distortion_coeff, np.ndarray)

            # Check inversed distortion
            img = mpimg.imread(calibration_folder.joinpath(file_name))
            cal_img = cam(img)
            self.assertEqual(img.shape, cal_img.shape)


if __name__ == '__main__':
    unittest.main()
