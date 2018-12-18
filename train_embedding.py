from src import extraction
from src import detection
from src import cropping
from src import embedding
from src import standardization
from scipy import misc
from src.embedding.train import train
from src import process
import numpy as np
import os
from pprint import pprint

input_path = '/Users/noverish/Desktop/datasets/children-rekognition-182'

if __name__ == '__main__':

    # image_paths, labels, class_list = process.process_extraction(input_path, crop=100)
    # image_paths, labels, rect_list = process.process_detection(image_paths, labels)
    # cropped_list = process.process_cropping(image_paths, rect_list, img_size=182, margin=44)

    train()

# def save_image(image, path):
#     dir_path = os.path.split(path)[0]
#
#     if not os.path.isdir(dir_path):
#         os.makedirs(dir_path)
#
#     misc.imsave(path, image)


# if __name__ == '__main__':
#     input_path = './train_inputs'
#     output_path = './outputs'
#
#     persons = extraction.extract(input_path)
#
#     for i, person in enumerate(persons):
#         for image_path in person['image_paths']:
#             faces = detection2.detect(image_path)
#
#             for j, face in enumerate(faces):
#                 cropped = cropping.crop(image_path, face, image_size, margin)
#
#                 person_path, file_name = os.path.split(image_path)
#                 file_name, file_ext = os.path.splitext(file_name)
#                 output_file_name = '{}_{}{}'.format(file_name, j, file_ext)
#                 output_file_path = os.path.join(output_path, 'crop', person['name'], output_file_name)
#                 output_dir_path, _ = os.path.split(output_file_path)
#
#                 if not os.path.isdir(output_dir_path):
#                     os.makedirs(output_dir_path)
#
#                 misc.imsave(output_file_path, cropped)
#
#         print(person['name'], 'crop done', '{}/{}'.format(i, len(persons)))
#
#     persons = extraction.extract('./outputs/crop/')
#
#     images = []
#
#     for person in persons:
#         for image_path in person['image_paths']:
#             image = misc.imread(image_path)
#
#             assert(image.shape == (image_size, image_size, 3))
#
#             images.append(image)
#
#     image_num = len(images)
#
#     images = np.array(images)
#
#     assert(images.shape == (image_num, image_size, image_size, 3))
#
#     images = standardization.standardize(images)
#
#     assert(images.shape == (image_num, image_size, image_size, 3))
#
#     images = images[:, :160, :160, ]
#
#     asdf = embedding.embed(images)
#
#     print(asdf)
#
#     model_path = os.path.join(os.path.split(__file__)[0], 'model/20180408-102900.pb')
#     print(model_path)
#
#     a = asdf[0]

    # # extract total image paths
    # image_classes = []
    #
    # input_path = os.path.abspath('./inputs')
    #
    #



    # for root, dirs, files in os.walk('./inputs'):
    #     for file in files:
    #         ext = os.path.splitext(__file__)[1]
    #
    #         if ext != '.jpg' and ext != '.png':
    #             continue
    #
    #         image_paths.append(os.path.join(root, file))
    #
    # print(image_paths)
    #
    # faces = detection2.detect(image_paths)
    #
    # print(faces)
