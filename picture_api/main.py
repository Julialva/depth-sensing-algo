from utils.image_utils import get_depth, show_image

def main():
    image = get_depth()
    show_image(image)

if __name__ == "__main__":
    main()