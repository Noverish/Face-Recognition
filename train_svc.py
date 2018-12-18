from src import clustering
from src import util
from src import process
import numpy as np

input_path = '/Users/noverish/Desktop/datasets/children-rekognition-182'
preprocessed_path = 'preprocessed/20181218-035825.pkl'
train_size = 0.8
random_seed = 666

if __name__ == '__main__':
    np.random.seed(random_seed)

    if util.is_file_exist(preprocessed_path):
        image_paths, labels, class_list, embedded_list = util.load_pickle(preprocessed_path)
        print('Successfully loaded pickle')
    else:
        preprocessed_path = 'preprocessed/' + util.get_curr_timestamp() + '.pkl'

        image_paths, labels, class_list = process.process_extraction(input_path)
        image_paths, labels, rect_list = process.process_detection(image_paths, labels)
        cropped_list = process.process_cropping(image_paths, rect_list)
        standardized_list = process.process_standardization(cropped_list)
        embedded_list = process.process_embedding(standardized_list)

        util.save_as_pickle(preprocessed_path, (image_paths, labels, class_list, embedded_list))
        print('Saved as pickle -', preprocessed_path)

    image_num = len(image_paths)
    print('Image num is', image_num)

    # Split data set to train and test
    train_num = int(image_num * train_size)
    test_num = image_num - train_num
    print('Train num is', train_num)
    print('Test num is', test_num)

    train_indexes = np.random.choice(image_num, train_num, replace=False)
    test_indexes = np.setdiff1d(np.arange(image_num), train_indexes)

    assert train_indexes.shape[0] == train_num
    assert test_indexes.shape[0] == test_num

    train_image_paths = [image_paths[i] for i in range(image_num) if i in train_indexes]
    test_image_paths = [image_paths[i] for i in range(image_num) if i in test_indexes]
    train_labels = [labels[i] for i in range(image_num) if i in train_indexes]
    test_labels = [labels[i] for i in range(image_num) if i in test_indexes]
    train_embeddings = embedded_list[train_indexes]
    test_embeddings = embedded_list[test_indexes]

    assert len(train_image_paths) == train_num
    assert len(test_image_paths) == test_num
    assert len(train_labels) == train_num
    assert len(test_labels) == test_num
    assert train_embeddings.shape[0] == train_num
    assert test_embeddings.shape[0] == test_num

    # train
    train_label_indexes = np.array([class_list.index(x) for x in train_labels])
    clustering.train(train_embeddings, train_label_indexes, class_list)
    print('Train done!!!')

    # evaluate
    test_label_indexes = np.array([class_list.index(x) for x in test_labels])
    score = clustering.evaluate(test_embeddings, test_label_indexes)
    print('score is ', score)
