import src.extraction as extraction
import src.detection as detection
import src.cropping as cropping
import src.standardization as standardization
import src.embedding as embedding
import src.clustering as clustering


def process_extraction(path, crop=-1):
    image_paths, labels, class_list = extraction.extract(path)

    if crop != -1:
        image_paths = image_paths[:crop]
        labels = labels[:crop]

    assert len(image_paths) == len(labels)
    print('Founded', str(len(image_paths)), 'images')

    return image_paths, labels, class_list


def process_detection(image_paths, labels):
    # Detect face from image list
    rect_list = detection.detect(image_paths)
    assert len(image_paths) == len(rect_list)

    # Remove no face images
    image_paths = [image_paths[i] for i in range(len(image_paths)) if rect_list[i] is not None]
    labels = [labels[i] for i in range(len(labels)) if rect_list[i] is not None]
    rect_list = [rect_list[i] for i in range(len(rect_list)) if rect_list[i] is not None]
    assert len(image_paths) == len(labels)
    assert len(image_paths) == len(rect_list)
    assert None not in rect_list
    print('Detected face in', str(len(rect_list)), 'images')

    return image_paths, labels, rect_list


def process_cropping(image_paths, rect_list, img_size=160, margin=0):
    # Crop face from images
    cropped_list = cropping.crop(image_paths, rect_list, img_size, margin)
    assert cropped_list.shape == (len(image_paths), img_size, img_size, 3)
    print('Cropped')

    return cropped_list


def process_standardization(cropped_list):
    # Standardize face images
    standardized_list = standardization.standardize(cropped_list)
    assert cropped_list.shape == (cropped_list.shape[0], 160, 160, 3)
    print('Standardized')

    return standardized_list


def process_embedding(standardized_list):
    # Embedded face images
    embedded_list = embedding.embed(standardized_list)
    assert embedded_list.shape == (embedded_list.shape[0], 512)
    print('Embedded')

    return embedded_list


def process_clustering(embedded_list):
    # Cluster
    predicted_list = clustering.predict(embedded_list)
    assert len(predicted_list) == embedded_list.shape[0]
    print('Clustered')

    return predicted_list
