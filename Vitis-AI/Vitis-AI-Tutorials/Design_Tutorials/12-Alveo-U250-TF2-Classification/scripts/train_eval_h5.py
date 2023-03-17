# Copyright 2021 Xilinx Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os, time
import numpy as np
import tensorflow as tf

from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.layers import Input
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions
from tensorflow.compat.v1 import flags
from tensorflow.keras.optimizers import RMSprop
from dataset import synth_input_fn
from dataset import input_fn, NUM_IMAGES
from dataset import get_images_infor_from_file, ImagenetSequence

keras = tf.keras

flags.DEFINE_string(
    'model', './train_dir/resnet50_model_195.h5',
    'TensorFlow \'GraphDef\' file to load.')
flags.DEFINE_bool(
    'eval_tfrecords', True,
    'If True then use tf_records data .')
flags.DEFINE_string(
    'data_dir', '/data3/datasets/Kaggle/fruits-360/tf_records',
    'The directory where put the eval images')
flags.DEFINE_bool(
    'eval_images', False,
    'If True then use tf_records data .')
flags.DEFINE_string(
    'eval_image_path', '/data3/datasets/Kaggle/fruits-360/val_for_tf2',
    'The directory where put the eval images')
flags.DEFINE_string(
    'eval_image_list',  '/data3/datasets/Kaggle/fruits-360/val_labels.txt', 'file has validation images list')
flags.DEFINE_string(
    'save_path', "train_dir",
    'The directory where save model')
flags.DEFINE_string(
    'filename', "resnet50_model_{epoch}.h5",
    'The name of sved model')
flags.DEFINE_integer(
    'label_offset', 1, 'label offset')
flags.DEFINE_string(
    'gpus', '0',
    'The gpus used for running evaluation.')
flags.DEFINE_bool(
    'eval_only', False,
    'If True then do not train model, only eval model.')
flags.DEFINE_bool(
    'save_whole_model', False,
    'as applications h5 file just include weights if true save whole model to h5 file.')
flags.DEFINE_bool(
    'use_synth_data', False,
    'If True then use synth data other than imagenet.')
flags.DEFINE_bool(
    'save_best_only', False,
    'If True then only save a model if `val_loss` has improved..')
flags.DEFINE_integer('train_step', None, 'Train step number')
flags.DEFINE_integer('batch_size', 32, 'Train batch size')
flags.DEFINE_integer('epochs', 200, 'Train epochs')
flags.DEFINE_integer('eval_batch_size', 50, 'Evaluate batch size')
flags.DEFINE_integer('save_every_epoch', 1, 'save every step number')
flags.DEFINE_integer('eval_every_epoch', 1, 'eval every step number')
flags.DEFINE_integer('steps_per_epoch', None, 'steps_per_epoch')
flags.DEFINE_integer('decay_steps', 10000, 'decay_steps')
flags.DEFINE_float('learning_rate', 1e-6, 'learning rate')
flags.DEFINE_bool('createnewmodel', False, 'Create a new model from the base Resnet50 model')
# Quantization Config
flags.DEFINE_bool('quantize', False, 'Whether to do quantization.')
flags.DEFINE_string('quantize_output_dir', './quantized/', 'Directory for quantize output results.')
flags.DEFINE_bool('quantize_eval', False, 'Whether to do quantize evaluation.')
flags.DEFINE_bool('dump', False, 'Whether to do dump.')
flags.DEFINE_string('dump_output_dir', './quantized/', 'Directory for dump output results.')

FLAGS = flags.FLAGS

TRAIN_NUM = NUM_IMAGES['train']
EVAL_NUM = NUM_IMAGES['validation']

def get_input_data(num_epochs=1):
  train_data = input_fn(
      is_training=True, data_dir=FLAGS.data_dir,
      batch_size=FLAGS.batch_size,
      num_epochs=num_epochs,
      num_gpus=1,
      dtype=tf.float32)

  eval_data = input_fn(
      is_training=False, data_dir=FLAGS.data_dir,
      batch_size=FLAGS.eval_batch_size,
      num_epochs=1,
      num_gpus=1,
      dtype=tf.float32)
  return train_data, eval_data


def main():
  ## run once to save h5 file (add model info)
  if FLAGS.save_whole_model:
    model = ResNet50(weights='imagenet')
    model.save(FLAGS.model)
    exit()

  if not FLAGS.eval_images:
    train_data, eval_data = get_input_data(FLAGS.epochs)

  if FLAGS.dump or FLAGS.quantize_eval:
      from tensorflow_model_optimization.quantization.keras import vitis_quantize
      with vitis_quantize.quantize_scope():
          model = keras.models.load_model(FLAGS.model)

  elif FLAGS.createnewmodel:
      #for training the model from scratch use the following:
      basemodel = ResNet50(weights='imagenet', include_top=True,input_tensor=Input(shape=(100, 100, 3)))
      base_output = basemodel.layers[175].output 
      new_output = tf.keras.layers.Dense(activation="softmax", units=131)(base_output)
      model = tf.keras.models.Model(inputs=basemodel.inputs, outputs=new_output)
      print(model.summary())

  else:
      model = keras.models.load_model(FLAGS.model)
      print(model.summary())

  img_paths, labels = get_images_infor_from_file(FLAGS.eval_image_path,
          FLAGS.eval_image_list, FLAGS.label_offset)
  imagenet_seq = ImagenetSequence(img_paths[0:1000], labels[0:1000], FLAGS.eval_batch_size)

  if FLAGS.quantize:
      # do quantization
      from tensorflow_model_optimization.quantization.keras import vitis_quantize
      #model = vitis_quantize.VitisQuantizer(model).quantize_model(calib_dataset=imagenet_seq)
      model = vitis_quantize.VitisQuantizer(model).quantize_model(calib_dataset=eval_data)

      # save quantized model
      model.save(os.path.join(FLAGS.quantize_output_dir, 'quantized.h5'))
      print('Quantize finished, results in: {}'.format(FLAGS.quantize_output_dir))
      return

  img_paths, labels = get_images_infor_from_file(FLAGS.eval_image_path,
          FLAGS.eval_image_list, FLAGS.label_offset)
  imagenet_seq = ImagenetSequence(img_paths[0:1], labels[0:1], FLAGS.eval_batch_size)

  if FLAGS.dump:
      # do quantize dump
      quantizer = vitis_quantize.VitisQuantizer.dump_model(model, imagenet_seq, FLAGS.dump_output_dir)

      print('Dump finished, results in: {}'.format(FLAGS.dump_output_dir))
      return

  initial_learning_rate = FLAGS.learning_rate
  lr_schedule = keras.optimizers.schedules.ExponentialDecay(
              initial_learning_rate, decay_steps=FLAGS.decay_steps, decay_rate=0.96,
              staircase=True

          )
  opt = RMSprop(learning_rate=lr_schedule)
  
  loss = keras.losses.SparseCategoricalCrossentropy()
  metric_top_5 = keras.metrics.SparseTopKCategoricalAccuracy()
  accuracy = keras.metrics.SparseCategoricalAccuracy()
  model.compile(optimizer=opt, loss=loss,
          metrics=[accuracy, metric_top_5])
  if not FLAGS.eval_only:
    if not os.path.exists(FLAGS.save_path):
      os.makedirs(FLAGS.save_path)
    callbacks = [
      keras.callbacks.ModelCheckpoint(
          filepath=os.path.join(FLAGS.save_path,FLAGS.filename),
          save_best_only=True,
          monitor="sparse_categorical_accuracy",
          verbose=1,
      )]
    steps_per_epoch = FLAGS.steps_per_epoch if FLAGS.steps_per_epoch else np.ceil(TRAIN_NUM/FLAGS.batch_size)
    model.fit(train_data,
            epochs=FLAGS.epochs,
            callbacks=callbacks,
            steps_per_epoch=steps_per_epoch,
            validation_freq=FLAGS.eval_every_epoch,
            validation_steps = EVAL_NUM/FLAGS.eval_batch_size,
            validation_data=eval_data)
  if not FLAGS.eval_images:
    print("evaluate model using tf_records data format")
    model.evaluate(eval_data, steps=EVAL_NUM/FLAGS.eval_batch_size)
  if FLAGS.eval_images and FLAGS.eval_only:
    img_paths, labels = get_images_infor_from_file(FLAGS.eval_image_path,
            FLAGS.eval_image_list, FLAGS.label_offset)
    imagenet_seq = ImagenetSequence(img_paths, labels, FLAGS.eval_batch_size)
    res = model.evaluate(imagenet_seq, steps=EVAL_NUM/FLAGS.eval_batch_size, verbose=1)


if __name__ == "__main__":
  os.environ['CUDA_VISIBLE_DEVICES'] = FLAGS.gpus
  main()
