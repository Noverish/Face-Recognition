from src import process
import os

input_path = '/Users/noverish/Desktop/children-pictures/age5/10510'

if __name__ == '__main__':
    image_paths = [os.path.join(input_path, x) for x in os.listdir(input_path)]

    image_paths, labels, rect_list = process.process_detection(image_paths, image_paths)
    cropped_list = process.process_cropping(image_paths, rect_list)
    image_paths, labels, augmented_list = process.process_augmentation(image_paths, labels, cropped_list)
    aligned_list = process.process_alignment(augmented_list)
    standardized_list = process.process_standardization(aligned_list)
    embedded_list = process.process_embedding(standardized_list)
    predicted_list = process.process_clustering(embedded_list)

    print(predicted_list)
