import os


def extract(input_path):
    input_path = os.path.abspath(input_path)

    image_paths = []
    labels = []

    person_names = sorted([x for x in os.listdir(input_path) if os.path.isdir(os.path.join(input_path, x))])

    for i in range(len(person_names)):
        person_name = person_names[i]
        person_path = os.path.join(input_path, person_name)

        image_names = sorted([x for x in os.listdir(person_path) if os.path.isfile(os.path.join(person_path, x))])

        for image_name in image_names:
            image_path = os.path.join(person_path, image_name)
            ext = os.path.splitext(image_path)[1].lower()

            if ext in ['.jpg', '.png']:
                image_paths.append(image_path)
                labels.append(person_name)

    return image_paths, labels, person_names
