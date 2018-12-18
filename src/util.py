from datetime import datetime
import time
import os
import pickle
import json


# Time
def get_curr_timestamp():
    return datetime.strftime(datetime.now(), '%Y%m%d-%H%M%S')


def get_curr_millis():
    return int(round(time.time() * 1000))


# File System
def create_folder_if_not_exist(path):
    os.makedirs(path)


def is_folder_exist(path):
    if path is None:
        return False

    if type(path) is not str:
        return False

    if path == '':
        return False

    return os.path.isdir(path)


def is_file_exist(path):
    if path is None:
        return False

    if type(path) is not str:
        return False

    if path == '':
        return False

    return os.path.isfile(path)


# Pickle
def save_as_pickle(path, obj):
    with open(path, 'wb') as f:
        pickle.dump(obj, f)


def load_pickle(path):
    with open(path, 'rb') as f:
        return pickle.load(f)


# json
def write_json(path, obj):
    file_name = datetime.strftime(datetime.now(), '%Y%m%d-%H%M%S.json')
    file_path = os.path.join(path, file_name)

    if not os.path.isdir(path):
        os.makedirs(path)

    with open(file_path, 'w') as f:
        f.write(json.dumps(obj))
