import os
import shutil
import json
from PIL import Image

path = os.path.expanduser('~/Desktop/children-pictures/age5')
path2 = os.path.expanduser('~/Desktop/age5_json')
margin = 44

i = 1

for root, dirs, files in os.walk(path2):
    for file in files:
        ext = file.split('.')[1]
        path = os.path.join(root, file)
        parent = os.path.abspath(os.path.join(path, os.pardir))

        if ext != 'json':
            continue

        image_path = path.replace('age5_json', 'children-pictures/age5')
        image_path = image_path.replace('.json', '.jpg')

        with open(path, 'r') as f:
            faces = json.loads(f.read())

        if len(faces) == 0:
            continue

        image = Image.open(image_path)

        for face in faces:
            name = face['name']

            if name is None:
                continue

            left = face['left']
            top = face['top']
            width = face['width']
            height = face['height']
            right = left + width
            bottom = top + height

            if width < 50 or height < 50:
                continue

            left = max(left - margin, 0)
            top = max(top - margin, 0)
            right = min(right + margin, image.size[0])
            bottom = min(bottom + margin, image.size[1])

            rect = (left, top, right, bottom)
            cropped = image.crop(rect)
            resized = cropped.resize((182, 182), Image.BILINEAR)

            dest_path = os.path.expanduser('~/Desktop/train2')
            dest_path = os.path.join(dest_path, name, str(i) + '.jpg')
            dest_parent = os.path.abspath(os.path.join(dest_path, os.pardir))

            if not os.path.isdir(dest_parent):
                os.makedirs(dest_parent)

            resized.save(dest_path, 'JPEG')

            i += 1

            print(i)
