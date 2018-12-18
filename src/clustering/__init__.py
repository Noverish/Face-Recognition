import os
import pickle
import numpy as np
from sklearn.svm import SVC


model_path = os.path.join(os.path.split(__file__)[0], '../../models/clustering/model.pkl')


def train(embeddings, labels, class_list):
    model = SVC(kernel='linear', probability=True)
    model.fit(embeddings, labels)

    with open(model_path, 'wb') as f:
        pickle.dump((model, labels, class_list), f)


def predict(embeddings):
    with open(model_path, 'rb') as f:
        (model, labels, class_list) = pickle.load(f)

    predictions = model.predict_proba(embeddings)
    best_class_indices = np.argmax(predictions, axis=1)
    return [class_list[x] for x in best_class_indices]
    # best_class_probabilities = predictions[np.arange(len(best_class_indices)), best_class_indices]
    #
    # labels = []
    # for i in range(len(best_class_indices)):
    #     labels.append(class_list[best_class_indices[i]])
    #
    # return labels, best_class_probabilities


def evaluate(embeddings, label_indexes):
    with open(model_path, 'rb') as f:
        (model, _, class_list) = pickle.load(f)

    return model.score(embeddings, label_indexes)
