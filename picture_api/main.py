from utils.image_utils import get_depth, show_image, capture_mono_image
import os
import time

def main():
    image = get_depth()
    show_image(image)

def capture_image_chunk(size=10):
    os.chdir('..')
    path = os.getcwd()
    for _ in range(size):
        timestamp = int(time.time())
        capture_mono_image(path, timestamp)

if __name__ == "__main__":
    main()
capture_image_chunk()
