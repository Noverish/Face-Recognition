from src import detection2
import os

if __name__ == '__main__':

    # extract total image paths
    image_paths = []
    for root, dirs, files in os.walk('./inputs'):
        for file in files:
            ext = os.path.splitext(__file__)[1]

            if ext != '.jpg' and ext != '.png':
                continue

            image_paths.append(os.path.join(root, file))

    faces = detection2.detect(image_paths)

    print(faces)
