from datetime import datetime
import os
import numpy as np
import tensorflow as tf
import src.embedding.inception_resnet_v1 as network
import src.embedding.facenet as facenet
from tensorflow.python.ops import data_flow_ops
import time
import itertools

args = {
    'batch_size': 2,
    'epoch_size': 2,
    'max_nrof_epochs': 2,

    'data_dir': '/Users/noverish/Desktop/datasets/children-rekognition-182',
    'learning_rate_schedule_file': 'data/learning_rate_schedule.txt',
    'logs_base_dir': '/Users/noverish/Desktop/aaa/logs/',
    'models_base_dir': '/Users/noverish/Desktop/aaa/models/',

    'alpha': 0.2,
    'embedding_size': 128,
    'gpu_memory_fraction': 1.0,
    'image_size': 160,
    'images_per_person': 40,
    'keep_probability': 1.0,
    'learning_rate': 0.01,
    'learning_rate_decay_epochs': 100,
    'learning_rate_decay_factor': 1.0,
    'moving_average_decay': 0.9999,
    'optimizer': 'RMSPROP',
    'people_per_batch': 6,
    'pretrained_model': '',
    'random_crop': False,
    'random_flip': False,
    'seed': 666,
    'weight_decay': 0.0001
}


def train():
    subdir = datetime.strftime(datetime.now(), '%Y%m%d-%H%M%S')
    log_dir = os.path.join(os.path.expanduser(args['logs_base_dir']), subdir)
    model_dir = os.path.join(os.path.expanduser(args['models_base_dir']), subdir)

    if not os.path.isdir(log_dir):
        os.makedirs(log_dir)
    if not os.path.isdir(model_dir):
        os.makedirs(model_dir)

    np.random.seed(seed=args['seed'])
    train_set = facenet.get_dataset(args['data_dir'])

    print('Model directory: %s' % model_dir)
    print('Log directory: %s' % log_dir)
    if args['pretrained_model']:
        print('Pre-trained model: %s' % os.path.expanduser(args['pretrained_model']))

    with tf.Graph().as_default():
        tf.set_random_seed(args['seed'])
        global_step = tf.Variable(0, trainable=False)

        learning_rate_placeholder = tf.placeholder(tf.float32, name='learning_rate')
        batch_size_placeholder = tf.placeholder(tf.int32, name='batch_size')
        phase_train_placeholder = tf.placeholder(tf.bool, name='phase_train')
        image_paths_placeholder = tf.placeholder(tf.string, shape=(None, 3), name='image_paths')
        labels_placeholder = tf.placeholder(tf.int64, shape=(None, 3), name='labels')

        input_queue = data_flow_ops.FIFOQueue(capacity=100000,
                                              dtypes=[tf.string, tf.int64],
                                              shapes=[(3,), (3,)],
                                              shared_name=None, name=None)

        enqueue_op = input_queue.enqueue_many([image_paths_placeholder, labels_placeholder])

        nrof_preprocess_threads = 4
        images_and_labels = []
        for _ in range(nrof_preprocess_threads):
            filenames, label = input_queue.dequeue()
            images = []
            for filename in tf.unstack(filenames):
                file_contents = tf.read_file(filename)
                image = tf.image.decode_image(file_contents, channels=3)

                if args['random_crop']:
                    image = tf.random_crop(image, [args['image_size'], args['image_size'], 3])
                else:
                    image = tf.image.resize_image_with_crop_or_pad(image, args['image_size'], args['image_size'])

                if args['random_flip']:
                    image = tf.image.random_flip_left_right(image)

                image.set_shape((args['image_size'], args['image_size'], 3))
                images.append(tf.image.per_image_standardization(image))
            images_and_labels.append([images, label])

        image_batch, labels_batch = tf.train.batch_join(images_and_labels,
                                                        batch_size=batch_size_placeholder,
                                                        shapes=[(args['image_size'], args['image_size'], 3), ()],
                                                        enqueue_many=True,
                                                        capacity=4 * nrof_preprocess_threads * args['batch_size'],
                                                        allow_smaller_final_batch=True)
        image_batch = tf.identity(image_batch, 'image_batch')
        image_batch = tf.identity(image_batch, 'input')
        labels_batch = tf.identity(labels_batch, 'label_batch')

        prelogits, _ = network.inference(image_batch,
                                         args['keep_probability'],
                                         phase_train=phase_train_placeholder,
                                         bottleneck_layer_size=args['embedding_size'],
                                         weight_decay=args['weight_decay'])

        embeddings = tf.nn.l2_normalize(prelogits, 1, 1e-10, name='embeddings')
        print('embeddings', str(embeddings))

        anchor, positive, negative = tf.unstack(tf.reshape(embeddings, [-1, 3, args['embedding_size']]), 3, 1)
        triplet_loss = facenet.triplet_loss(anchor, positive, negative, args['alpha'])

        learning_rate = tf.train.exponential_decay(learning_rate_placeholder,
                                                   global_step,
                                                   args['learning_rate_decay_epochs'] * args['epoch_size'],
                                                   args['learning_rate_decay_factor'],
                                                   staircase=True)
        tf.summary.scalar('learning_rate', learning_rate)

        regularization_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
        total_loss = tf.add_n([triplet_loss] + regularization_losses, name='total_loss')

        train_op = facenet.train(total_loss,
                                 global_step,
                                 args['optimizer'],
                                 learning_rate,
                                 args['moving_average_decay'],
                                 tf.global_variables())

        saver = tf.train.Saver(tf.trainable_variables(), max_to_keep=3)

        summary_op = tf.summary.merge_all()

        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=args['gpu_memory_fraction'])
        sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

        sess.run(tf.global_variables_initializer(), feed_dict={phase_train_placeholder: True})
        sess.run(tf.local_variables_initializer(), feed_dict={phase_train_placeholder: True})

        summary_writer = tf.summary.FileWriter(log_dir, sess.graph)
        coord = tf.train.Coordinator()
        tf.train.start_queue_runners(coord=coord, sess=sess)

        with sess.as_default():

            if args['pretrained_model']:
                print('Restoring pretrained model: %s' % args['pretrained_model'])
                saver.restore(sess, os.path.expanduser(args['pretrained_model']))

            epoch = 0
            while epoch < args['max_nrof_epochs']:
                step = sess.run(global_step, feed_dict=None)
                epoch = step // args['epoch_size']

                _train(args, sess,
                       dataset=train_set,
                       epoch=epoch,
                       image_paths_placeholder=image_paths_placeholder,
                       labels_placeholder=labels_placeholder,
                       labels_batch=labels_batch,
                       batch_size_placeholder=batch_size_placeholder,
                       learning_rate_placeholder=learning_rate_placeholder,
                       phase_train_placeholder=phase_train_placeholder,
                       enqueue_op=enqueue_op,
                       input_queue=input_queue,
                       global_step=global_step,
                       embeddings=embeddings,
                       loss=total_loss,
                       train_op=train_op,
                       summary_op=summary_op,
                       summary_writer=summary_writer,
                       learning_rate_schedule_file=args['learning_rate_schedule_file'],
                       embedding_size=args['embedding_size'],
                       anchor=anchor,
                       positive=positive,
                       negative=negative,
                       triplet_loss=triplet_loss)

                _save_variables_and_metagraph(sess, saver, summary_writer, model_dir, subdir, step)

    return model_dir


def _train(args, sess, dataset, epoch, image_paths_placeholder, labels_placeholder, labels_batch,
           batch_size_placeholder, learning_rate_placeholder, phase_train_placeholder, enqueue_op, input_queue,
           global_step,
           embeddings, loss, train_op, summary_op, summary_writer, learning_rate_schedule_file,
           embedding_size, anchor, positive, negative, triplet_loss):
    batch_number = 0

    if args['learning_rate'] > 0.0:
        lr = args['learning_rate']
    else:
        lr = facenet.get_learning_rate_from_file(learning_rate_schedule_file, epoch)

    while batch_number < args['epoch_size']:
        image_paths, num_per_class = _sample_people(dataset, args['people_per_batch'], args['images_per_person'])

        print('Running forward pass on sampled images: ', end='')
        start_time = time.time()
        nrof_examples = args['people_per_batch'] * args['images_per_person']
        labels_array = np.reshape(np.arange(nrof_examples), (-1, 3))
        image_paths_array = np.reshape(np.expand_dims(np.array(image_paths), 1), (-1, 3))
        sess.run(enqueue_op, {image_paths_placeholder: image_paths_array, labels_placeholder: labels_array})
        emb_array = np.zeros((nrof_examples, embedding_size))
        nrof_batches = int(np.ceil(nrof_examples / args['batch_size']))
        for i in range(nrof_batches):
            batch_size = min(nrof_examples - i * args['batch_size'], args['batch_size'])
            emb, lab = sess.run([embeddings, labels_batch], feed_dict={batch_size_placeholder: batch_size,
                                                                       learning_rate_placeholder: lr,
                                                                       phase_train_placeholder: True})
            emb_array[lab, :] = emb
        print('%.3f' % (time.time() - start_time))

        print('Selecting suitable triplets for training')
        triplets, nrof_random_negs, nrof_triplets = _select_triplets(emb_array,
                                                                     num_per_class,
                                                                     image_paths,
                                                                     args['people_per_batch'],
                                                                     args['alpha'])
        selection_time = time.time() - start_time
        print('(nrof_random_negs, nrof_triplets) = (%d, %d): time=%.3f seconds' %
              (nrof_random_negs, nrof_triplets, selection_time))

        nrof_batches = int(np.ceil(nrof_triplets * 3 / args['batch_size']))
        triplet_paths = list(itertools.chain(*triplets))
        labels_array = np.reshape(np.arange(len(triplet_paths)), (-1, 3))
        triplet_paths_array = np.reshape(np.expand_dims(np.array(triplet_paths), 1), (-1, 3))
        sess.run(enqueue_op, {image_paths_placeholder: triplet_paths_array, labels_placeholder: labels_array})
        nrof_examples = len(triplet_paths)
        train_time = 0
        i = 0
        emb_array = np.zeros((nrof_examples, embedding_size))
        loss_array = np.zeros((nrof_triplets,))
        summary = tf.Summary()
        step = 0
        while i < nrof_batches:
            start_time = time.time()
            batch_size = min(nrof_examples - i * args['batch_size'], args['batch_size'])
            feed_dict = {batch_size_placeholder: batch_size, learning_rate_placeholder: lr,
                         phase_train_placeholder: True}
            err, _, step, emb, lab = sess.run([loss, train_op, global_step, embeddings, labels_batch],
                                              feed_dict=feed_dict)
            emb_array[lab, :] = emb
            loss_array[i] = err
            duration = time.time() - start_time
            print('Epoch: [%d][%d/%d]\tTime %.3f\tLoss %2.3f' %
                  (epoch, batch_number + 1, args['epoch_size'], duration, err))
            batch_number += 1
            i += 1
            train_time += duration
            summary.value.add(tag='loss', simple_value=err)

        summary.value.add(tag='time/selection', simple_value=selection_time)
        summary_writer.add_summary(summary, step)
    return step


def _select_triplets(embeddings, nrof_images_per_class, image_paths, people_per_batch, alpha):
    trip_idx = 0
    emb_start_idx = 0
    num_trips = 0
    triplets = []

    for i in range(people_per_batch):
        nrof_images = int(nrof_images_per_class[i])
        for j in range(1, nrof_images):
            a_idx = emb_start_idx + j - 1
            neg_dists_sqr = np.sum(np.square(embeddings[a_idx] - embeddings), 1)
            for pair in range(j, nrof_images):
                p_idx = emb_start_idx + pair
                pos_dist_sqr = np.sum(np.square(embeddings[a_idx] - embeddings[p_idx]))
                neg_dists_sqr[emb_start_idx:emb_start_idx + nrof_images] = np.NaN

                all_neg = np.where(neg_dists_sqr - pos_dist_sqr < alpha)[0]
                nrof_random_negs = all_neg.shape[0]
                if nrof_random_negs > 0:
                    rnd_idx = np.random.randint(nrof_random_negs)
                    n_idx = all_neg[rnd_idx]
                    triplets.append((image_paths[a_idx], image_paths[p_idx], image_paths[n_idx]))

                    trip_idx += 1

                num_trips += 1

        emb_start_idx += nrof_images

    np.random.shuffle(triplets)
    return triplets, num_trips, len(triplets)


def _sample_people(dataset, people_per_batch, images_per_person):
    nrof_images = people_per_batch * images_per_person

    nrof_classes = len(dataset)
    class_indices = np.arange(nrof_classes)
    np.random.shuffle(class_indices)

    i = 0
    image_paths = []
    num_per_class = []
    sampled_class_indices = []

    while len(image_paths) < nrof_images:
        class_index = class_indices[i]
        nrof_images_in_class = len(dataset[class_index])
        image_indices = np.arange(nrof_images_in_class)
        np.random.shuffle(image_indices)
        nrof_images_from_class = min(nrof_images_in_class, images_per_person, nrof_images - len(image_paths))
        idx = image_indices[0:nrof_images_from_class]
        image_paths_for_class = [dataset[class_index].image_paths[j] for j in idx]
        sampled_class_indices += [class_index] * nrof_images_from_class
        image_paths += image_paths_for_class
        num_per_class.append(nrof_images_from_class)
        i += 1

    return image_paths, num_per_class


def _save_variables_and_metagraph(sess, saver, summary_writer, model_dir, model_name, step):
    print('Saving variables')
    start_time = time.time()
    checkpoint_path = os.path.join(model_dir, 'model-%s.ckpt' % model_name)
    saver.save(sess, checkpoint_path, global_step=step, write_meta_graph=False)
    save_time_variables = time.time() - start_time
    print('Variables saved in %.2f seconds' % save_time_variables)
    metagraph_filename = os.path.join(model_dir, 'model-%s.meta' % model_name)
    save_time_metagraph = 0
    if not os.path.exists(metagraph_filename):
        print('Saving metagraph')
        start_time = time.time()
        saver.export_meta_graph(metagraph_filename)
        save_time_metagraph = time.time() - start_time
        print('Metagraph saved in %.2f seconds' % save_time_metagraph)
    summary = tf.Summary()

    summary.value.add(tag='time/save_variables', simple_value=save_time_variables)
    summary.value.add(tag='time/save_metagraph', simple_value=save_time_metagraph)
    summary_writer.add_summary(summary, step)
