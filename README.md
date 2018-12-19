# Face Recognition

### 1. install dependencies

```commandline
$ pip3 install -r requirements.txt
```

### 2. Download model files

[download link](https://drive.google.com/file/d/1t9t95FF3pS5sVqrk2yl1dr4HU6NtYIUB/view?usp=sharing)

Unzip and Place it to project folder (like below)

```
Face-Recognition
|-models
  |-alignment
    |- ...
  |-clustering
    |- ...
  |-detection
    |- ...
  |-embedding
    |- ...
|-src
|-predict.py
|-train_embedding.py
|-train_svc
...
```

### 3. Modify image folder path of predict.py

Change `input_path` in predict.py to your image folder path

image folder structure is below. (No subdir is allowed)

```
your-folder
|-some_image.jpg
|-another_image.jpg
|-other_image.jpg
...
```

### 4. Run

```commandline
$ python3 predict.py
```