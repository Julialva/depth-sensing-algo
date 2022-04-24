from utils.image_utils import get_point_cloud, show_image, capture_mono_image,dump_point_cloud,capture_image_chunk
import os
import time

def main():
    image = get_point_cloud()
    #show_image(image[:,:,3])
    dump_point_cloud(image,"point_cloud")

if __name__ == "__main__":
    main()
    capture_image_chunk()
