import numpy as np
import sys
import pyzed.sl as sl
import cv2 
import os
import pickle
from datetime import datetime,timezone

def capture_mono_image(camera_index: int):
    cap = cv2.VideoCapture(camera_index) # video capture source camera (Here webcam of laptop) 
    ret,frame = cap.read() # return a single frame in variable `frame`
    cap.release()
    return frame

def dump_mono_image(frame, directory_path: str, timestamp: int):
    path = f"{directory_path}/img_{timestamp}.png"
    cv2.imwrite(path,frame)

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

def get_point_cloud()-> np.ndarray:
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
        return "Error!"

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

def dump_point_cloud(point_cloud: np.ndarray, path: str, timestamp: int) -> None:

    filename = f"{path}/point_cloud_{timestamp}"
    with open(filename,'wb') as f:
        pickle.dump(point_cloud, f)
    return

def read_point_cloud(path: str):
    with open(path,'rb') as f: 
        arr = pickle.load(f)
    return arr

def capture_image_batch(batch_size: int, path_mono_img: str,path_stereo: str, mono_camera_index: int):
    capture_arr = []
    for _ in range(batch_size):
        timestamp = int(datetime.now(tz=timezone.utc).timestamp())
        frame = capture_mono_image(mono_camera_index)
        dump_mono_image(frame, path_mono_img, timestamp)
        point_cloud = get_point_cloud()

        if isinstance(point_cloud,str):
           capture_arr.append("Error!") # point cloud error msg
        else:    
            dump_point_cloud(point_cloud,path_stereo,timestamp)
            capture_arr.append("OK!")
    return "Done!", capture_arr

def list_ports():
    """
    Test the ports and returns a tuple with the available ports and the ones that are working.
    """
    non_working_ports = []
    dev_port = 0
    working_ports = []
    available_ports = []
    while len(non_working_ports) < 6: # if there are more than 5 non working ports stop the testing. 
        camera = cv2.VideoCapture(dev_port)
        if not camera.isOpened():
            non_working_ports.append(dev_port)
            print("Port %s is not working." %dev_port)
        else:
            is_reading, img = camera.read()
            w = camera.get(3)
            h = camera.get(4)
            if is_reading:
                print("Port %s is working and reads images (%s x %s)" %(dev_port,h,w))
                working_ports.append(dev_port)
            else:
                print("Port %s for camera ( %s x %s) is present but does not reads." %(dev_port,h,w))
                available_ports.append(dev_port)
        dev_port +=1
    return available_ports,working_ports,non_working_ports