import simpai
import numpy as np

if __name__ == '__main__':
    test_jpg = simpai.data.filepath_to_ndarray('./test/test.jpg', transpose = 'hwc', enhancement_enable = False)
    test_png = simpai.data.filepath_to_ndarray('./test/test.png', transpose = 'chw', enhancement_enable = False)

    simpai.vis.show_hwc_ndarray(test_jpg)
    simpai.vis.show_chw_ndarray(test_png)

    gray_img = np.random.rand(300, 400)
    simpai.vis.show_hw_ndarray(gray_img)
