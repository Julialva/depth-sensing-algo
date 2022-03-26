import numpy as np
import sys
import pyzed.sl as sl

def show_image(image: np.ndarray, title: str = "Image", cmap_type: str = "gray"):
    """Display a image. This is ment for development use only

    :param image: Array containing image data
    :type image: ndarray
    :param title: Image title, defaults to "Image"
    :type title: str, optional
    :param cmap_type: color map, defaults to "gray"
    :type cmap_type: str, optional
    """
    import matplotlib.pyplot as plt
    plt.imshow(image, cmap=cmap_type)
    plt.title(title)
    plt.axis("off")
    plt.show()

def get_depth()->np.ndarray:
    """Retrieves point cloud from zed camera. 
    This function requires a zed camera to be available to the machine.

    :return: point cloud array
    :rtype: np.ndarray
    """
    init = sl.InitParameters(camera_resolution=sl.RESOLUTION.HD2K,
                                 depth_mode=sl.DEPTH_MODE.ULTRA,
                                 coordinate_units=sl.UNIT.METER,
                                 coordinate_system=sl.COORDINATE_SYSTEM.RIGHT_HANDED_Y_UP)
    init.depth_minimum_distance =0.2
    init.depth_maximum_distance=40

    zed = sl.Camera()
    status = zed.open(init)
    if status != sl.ERROR_CODE.SUCCESS:
        print(repr(status))
        exit()

    res = sl.Resolution()
    res.width = 2048
    res.height = 1080
    point_cloud = sl.Mat(res.width, res.height, sl.MAT_TYPE.F32_C4, sl.MEM.CPU)

    runtime_parameters =sl.RuntimeParameters()
    runtime_parameters.sensing_mode = sl.SENSING_MODE.FILL
    if zed.grab(runtime_parameters) == sl.ERROR_CODE.SUCCESS:
        zed.retrieve_measure(point_cloud, sl.MEASURE.XYZRGBA,sl.MEM.CPU, res)
    zed.close()
    return point_cloud.get_data()