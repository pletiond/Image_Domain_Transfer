import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np
import pickle
import os
import scipy.io
import scipy.misc
import imageio
import dill
from PIL import Image
from loader import Loader

def resize_images(image_arrays, size=[64, 64]):
    # convert float type to integer
    image_arrays = (image_arrays * 255).astype('uint8')

    resized_image_arrays = np.zeros([image_arrays.shape[0]] + size)
    for i, image_array in enumerate(image_arrays):
        image = Image.fromarray(image_array)
        resized_image = image.resize(size=size, resample=Image.ANTIALIAS)

        resized_image_arrays[i] = np.asarray(resized_image)

    return np.expand_dims(resized_image_arrays, 3)


class Solver(object):

    def __init__(self, model, batch_size=100, pretrain_iter=20000, train_iter=8000, sample_iter=10,
                 svhn_dir='svhn', mnist_dir='mnist', log_dir='logs', sample_save_path='sample', faces_dir = './wiki_dataset', emoji_dir = './emoji_dataset',
                 model_save_path='model', pretrained_model='model/svhn_model-10000', test_model='model/dtn-'):

        self.model = model
        self.batch_size = batch_size
        self.pretrain_iter = pretrain_iter
        self.train_iter = train_iter
        self.sample_iter = sample_iter
        self.svhn_dir = svhn_dir
        self.mnist_dir = mnist_dir
        self.faces_dir = faces_dir
        self.emoji_dir = emoji_dir
        self.log_dir = log_dir
        self.sample_save_path = sample_save_path
        self.model_save_path = model_save_path
        self.pretrained_model = pretrained_model
        self.test_model = test_model
        self.config = tf.ConfigProto()
        self.config.gpu_options.allow_growth = True
        self.loader = Loader(mode=self.model.mode)
        self.save_points = [500, 1000, 2000, 3000, 6000]

    def load_svhn(self, image_dir, split='train'):
        print('loading svhn image dataset..')

        if self.model.mode == 'pretrain':
            image_file = 'extra_32x32.mat' if split == 'train' else 'test_32x32.mat'
        else:
            image_file = 'train_32x32.mat' if split == 'train' else 'test_32x32.mat'

        image_dir = os.path.join(image_dir, image_file)
        svhn = scipy.io.loadmat(image_dir)

        images = np.transpose(svhn['X'], [3, 0, 1, 2]) / 127.5 - 1

        labels = svhn['y'].reshape(-1)
        labels[np.where(labels == 10)] = 0
        print('finished loading svhn image dataset..!')

        return images, labels

    def load_mnist(self, image_dir, split='train'):
        print('loading mnist image dataset..')
        image_file = 'train.pkl' if split == 'train' else 'test.pkl'
        image_dir = os.path.join(image_dir, image_file)
        with open(image_dir, 'rb') as f:
            mnist = pickle.load(f)
        images = mnist['X'] / 127.5 - 1
        labels = mnist['y']
        print('finished loading mnist image dataset..!')
        return images, labels

    def load_faces(self, image_dir):
        print('loading faces image dataset..')
        image_file = 'wiki_all.pkl'
        #image_dir = os.path.join(image_dir, image_file)
        with open(image_dir+ '/'+image_file, 'rb') as f:
            faces = pickle.load(f)

        print(np.array(faces).shape)
        images = np.array(faces) / 127.5 - 1

        print(f'Len faces: {len(images)}')
        print('finished loading faces image dataset..!')
        #print(images[0])
        return np.array(images)

    def load_emoji(self, image_dir):
        print('loading emoji image dataset..')
        image_file = 'emoji_all.pkl'
        #image_dir = os.path.join(image_dir, image_file)
        with open(image_dir+'/'+image_file, 'rb') as f:
            faces = pickle.load(f)
        images = np.array(faces) / 127.5 - 1
        #print(images[0])
        print(f'Len emojis: {len(images)}')
        print('finished loading emoji image dataset..!')
        return np.array(images)

    def load_clothes(self, split='train'):
        print('Loading clothes image dataset')
        if split == 'train':
            img_src = 'clothes_pretrain_img.dill'
            labels_src = 'clothes_pretrain_labels.dill'
        else:
            img_src = 'clothes_train_img.dill'
            labels_src = 'clothes_train_labels.dill'

        with open(img_src, 'rb') as img_file:
            images = dill.load(img_file)

        with open(labels_src, 'rb') as labels_file:
            labels = dill.load(labels_file)

        #images = np.array(images) / 127.5 - 1

        print('finished loading clothes image dataset..!')
        return np.array(images), np.array(labels)

    def load_fashion_mnist(self, split='train'):
        (train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.fashion_mnist.load_data()
        print(train_images.shape)
        print(test_images.shape)
        if split == 'train':
            return resize_images(train_images), train_labels
        else:
            return resize_images(test_images), test_labels


    def merge_images(self, sources, targets, k=10):
        _, h, w, _ = sources.shape
        row = int(np.sqrt(self.batch_size))
        merged = np.zeros([row * h, row * w * 2, 3])

        for idx, (s, t) in enumerate(zip(sources, targets)):
            i = idx // row
            j = idx % row
            merged[i * h:(i + 1) * h, (j * 2) * h:(j * 2 + 1) * h, :] = s
            merged[i * h:(i + 1) * h, (j * 2 + 1) * h:(j * 2 + 2) * h, :] = t
        return merged

    def pretrain(self):
        # load svhn dataset
        #train_images, train_labels = self.load_svhn(self.svhn_dir, split='train')
        #test_images, test_labels = self.load_svhn(self.svhn_dir, split='test')
        self.loader.load_fashion_image_data()

        #train_images, train_labels = self.load_clothes( split='train')
        #test_images, test_labels = self.load_clothes( split='test')

        # build a graph
        model = self.model
        model.build_model()

        print('Model was built!!')

        with tf.Session(config=self.config) as sess:
            tf.global_variables_initializer().run()
            saver = tf.train.Saver()
            summary_writer = tf.summary.FileWriter(logdir=self.log_dir, graph=tf.get_default_graph())

            for step in range(self.pretrain_iter):
                #i = step % int(train_images.shape[0] / self.batch_size)
                #batch_images = train_images[i * self.batch_size:(i + 1) * self.batch_size]  / 127.5 - 1
                #batch_labels = train_labels[i * self.batch_size:(i + 1) * self.batch_size]
                batch_images, batch_labels = self.loader.fashion_get_next_batch(batch_size=self.batch_size)
                #print(batch_labels)

                feed_dict = {model.images: np.array(batch_images), model.labels: np.array(batch_labels)}
                sess.run(model.train_op, feed_dict)



                if (step + 1) % 10 == 0:
                    test_images, test_labels = self.loader.fashion_get_next_batch(batch_size=self.batch_size)
                    summary, l, acc = sess.run([model.summary_op, model.loss, model.accuracy], feed_dict)
                    #rand_idxs = np.random.permutation(test_images.shape[0])[:self.batch_size]
                    test_acc, _ = sess.run(fetches=[model.accuracy, model.loss],
                                           feed_dict={model.images: test_images,#[rand_idxs]
                                                      model.labels: test_labels})
                    summary_writer.add_summary(summary, step)
                    print('Step: [%d/%d] loss: [%.6f] train acc: [%.2f] test acc [%.2f]' \
                          % (step + 1, self.pretrain_iter, l, acc, test_acc))

                if (step + 1) % 500 == 0:
                    saver.save(sess, os.path.join(self.model_save_path, 'svhn_model'), global_step=step + 1)
                    print('svhn_model-%d saved..!' % (step + 1))

    def train(self):
        # load datasets
        #source_images, _ = self.load_svhn(self.svhn_dir, split='train')

        #CLOTHES
        self.loader.load_fashion_image_data()
        self.loader.load_mnist_data()

        #source_images, _ = self.load_clothes(split='train')
        #target_images, _ = self.load_fashion_mnist(split='train')
        #target_images, _ = self.load_mnist(self.mnist_dir, split='train')

        # build a graph
        model = self.model
        model.build_model()

        # make directory if not exists
        if tf.gfile.Exists(self.log_dir):
            tf.gfile.DeleteRecursively(self.log_dir)
        tf.gfile.MakeDirs(self.log_dir)

        with tf.Session(config=self.config) as sess:
            # initialize G and D
            tf.global_variables_initializer().run()
            # restore variables of F
            print('loading pretrained model F..')
            variables_to_restore = slim.get_model_variables(scope='content_extractor')
            restorer = tf.train.Saver(variables_to_restore)
            restorer.restore(sess, self.pretrained_model)
            summary_writer = tf.summary.FileWriter(logdir=self.log_dir, graph=tf.get_default_graph())
            saver = tf.train.Saver()

            print('start training..!')
            f_interval = 15

            #  try range(int(source_images.shape[0] / self.batch_size))
            for step in range(self.train_iter + 1):

                i = step % int(self.loader.fashion_num_examples / self.batch_size)
                # train the model for source domain S
                #src_images = source_images[i * self.batch_size:(i + 1) * self.batch_size] / 127.5 - 1
                src_images, _ = self.loader.fashion_get_next_batch(batch_size=self.batch_size)
                feed_dict = {model.src_images: src_images}

                sess.run(model.d_train_op_src, feed_dict)
                sess.run([model.g_train_op_src], feed_dict)
                sess.run([model.g_train_op_src], feed_dict)
                sess.run([model.g_train_op_src], feed_dict)
                sess.run([model.g_train_op_src], feed_dict)
                sess.run([model.g_train_op_src], feed_dict)
                sess.run([model.g_train_op_src], feed_dict)

                if step > 3000:
                    f_interval = 30

                if i % f_interval == 0:
                    sess.run(model.f_train_op_src, feed_dict)

                if (step + 1) % 10 == 0:
                    summary, dl, gl, fl = sess.run([model.summary_op_src, \
                                                    model.d_loss_src, model.g_loss_src, model.f_loss_src], feed_dict)
                    summary_writer.add_summary(summary, step)
                    print('[Source] step: [%d/%d] d_loss: [%.6f] g_loss: [%.6f] f_loss: [%.6f]' \
                          % (step + 1, self.train_iter, dl, gl, fl))

                # train the model for target domain T
                #j = step % int(target_images.shape[0] / self.batch_size)
                #trg_images = target_images[j * self.batch_size:(j + 1) * self.batch_size] / 127.5 - 1
                trg_images, _ = self.loader.mnist_get_next_batch(self.batch_size)
                feed_dict = {model.src_images: src_images, model.trg_images: trg_images}
                sess.run(model.d_train_op_trg, feed_dict)
                sess.run(model.d_train_op_trg, feed_dict)
                sess.run(model.g_train_op_trg, feed_dict)
                sess.run(model.g_train_op_trg, feed_dict)
                sess.run(model.g_train_op_trg, feed_dict)
                sess.run(model.g_train_op_trg, feed_dict)

                if (step + 1) % 10 == 0:
                    summary, dl, gl = sess.run([model.summary_op_trg, \
                                                model.d_loss_trg, model.g_loss_trg], feed_dict)
                    summary_writer.add_summary(summary, step)
                    print('[Target] step: [%d/%d] d_loss: [%.6f] g_loss: [%.6f]' \
                          % (step + 1, self.train_iter, dl, gl))

                if (step + 1) in self.save_points:
                    saver.save(sess, os.path.join(self.model_save_path, 'dtn'), global_step=step + 1)
                    print('model/dtn-%d saved' % (step + 1))

    def eval(self):
        # build model
        model = self.model
        model.build_model()

        # load TEST!!!! dataset
        self.loader.load_fashion_image_data()
        #eval_dataset, _ = self.load_clothes(split='train')

        with tf.Session(config=self.config) as sess:
            # load trained parameters
            print('loading test model..')
            saver = tf.train.Saver()

            x = input('Enter checkpoint:')
            saver.restore(sess, self.test_model+x)

            print('start sampling..!')
            for i in range(self.sample_iter):
                # train model for source domain S
                #batch_images = eval_dataset[i * self.batch_size:(i + 1) * self.batch_size]
                batch_images, labels = self.loader.eval_fashion_get_next_batch(batch_size=self.batch_size)
                print(labels)
                feed_dict = {model.images: batch_images}
                sampled_batch_images = sess.run(model.sampled_images, feed_dict)

                # merge and save source images and sampled target images
                merged = self.merge_images(np.array(batch_images), sampled_batch_images)

                #with open(f'./sample/eval_src-{i}.dill', 'wb') as handle:
                #    dill.dump(batch_images, handle)

                #with open(f'./sample/eval_trg-{i}.dill', 'wb') as handle:
                #    dill.dump(sampled_batch_images, handle)

                #continue

                path = os.path.join(self.sample_save_path,
                                    str(x)+'-sample-%d-to-%d.png' % (i * self.batch_size, (i + 1) * self.batch_size))
                imageio.imwrite(path, merged)
                print('saved %s' % path)
