import os
import dlib
import openface


predictor_model = os.path.join(os.path.split(__file__)[0], '../../models/alignment/shape_predictor_68_face_landmarks.dat')


def align(images, verbose=True):
    for i in range(images.shape[0]):
        image = images[i]

        images[i] = align_one(image)

        if verbose:
            print('Aligning faces... ({}/{})'.format(i + 1, images.shape[0]))
    return images


def align_one(image):
    face_detector = dlib.get_frontal_face_detector()
    face_pose_predictor = dlib.shape_predictor(predictor_model)
    face_aligner = openface.AlignDlib(predictor_model)

    # Run the HOG face detector on the image data
    detected_faces = face_detector(image, 1)

    if len(detected_faces) == 0:
        return image

    # Loop through each face we found in the image
    for i, face_rect in enumerate(detected_faces):

        pose_landmarks = face_pose_predictor(image, face_rect)

        aligned_face = face_aligner.align(image.shape[0], image, face_rect, landmarkIndices=openface.AlignDlib.OUTER_EYES_AND_NOSE)

        return aligned_face
