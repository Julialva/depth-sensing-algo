import numpy as np
import cv2
import pickle
from datetime import datetime, timezone
from time import sleep


def capture_mono_image(camera_index: int):
    # video capture source camera (Here webcam of laptop)
    cap = cv2.VideoCapture(camera_index)
    ret, frame = cap.read()  # return a single frame in variable `frame`
    cap.release()
    return frame


def dump_mono_image(frame, directory_path: str, timestamp: int):
    path = f"{directory_path}/img_{timestamp}.png"
    cv2.imwrite(path, frame)


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


def dump_point_cloud(point_cloud: np.ndarray, path: str, timestamp: int) -> None:

    filename = f"{path}/point_cloud_{timestamp}"
    with open(filename, 'wb') as f:
        pickle.dump(point_cloud, f)
    return


def read_point_cloud(path: str):
    with open(path, 'rb') as f:
        arr = pickle.load(f)
    return arr


def capture_image_batch(batch_size: int,
                        path_left: str,
                        path_right: str,
                        mono_camera_index_left: int,
                        mono_camera_index_right: int):
    capture_arr = []
    for _ in range(batch_size):
        timestamp = int(datetime.now(tz=timezone.utc).timestamp())
        try:
            frame_left = capture_mono_image(mono_camera_index_left)
            sleep(0.5)
            frame_right = capture_mono_image(mono_camera_index_right)
            sleep(0.5)
            dump_mono_image(frame_left, path_left, timestamp)
            dump_mono_image(frame_right, path_right, timestamp)
        except:
            capture_arr.append("ERROR!") 
        else:  
            capture_arr.append("OK!")
    return "Done!", capture_arr


def list_ports():
    """
    Test the ports and returns a tuple with the available ports 
    and the ones that are working.
    """
    non_working_ports = []
    dev_port = 0
    working_ports = []
    available_ports = []
    # if there are more than 5 non working ports stop the testing.
    while len(non_working_ports) < 6:
        camera = cv2.VideoCapture(dev_port)
        if not camera.isOpened():
            non_working_ports.append(dev_port)
            print("Port %s is not working." % dev_port)
        else:
            is_reading, img = camera.read()
            w = camera.get(3)
            h = camera.get(4)
            if is_reading:
                print("Port %s is working and reads images (%s x %s)" %
                      (dev_port, h, w))
                working_ports.append(dev_port)
            else:
                print("Port % s for camera ( % s x % s)" +
                      "is present but does not reads." % (
                          dev_port, h, w))
                available_ports.append(dev_port)
        dev_port += 1
    return available_ports, working_ports, non_working_ports
