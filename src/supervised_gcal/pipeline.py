"""Trains and Evaluates the SupervisedGCAL  network using a feed dictionary."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# pylint: disable=missing-docstring
import argparse
import os.path
import sys
import time

import numpy as np
import tensorflow as tf
from six.moves import xrange  # pylint: disable=redefined-builtin
from tensorflow.examples.tutorials.mnist import input_data
from tensorflow.examples.tutorials.mnist import mnist
from tensorflow.python import debug as tf_debug

from tf_src.src import add_images_summaries
# Basic model parameters as external flags.
from tf_src.src import supervised_gcal_graph

FLAGS = None


def placeholder_inputs(batch_size):
    """Generate placeholder variables to represent the input tensors.

  These placeholders are used as inputs by the rest of the model building
  code and will be fed from the downloaded data in the .run() loop, below.

  Args:
    batch_size: The batch size will be baked into both placeholders.

  Returns:
    images_placeholder: Images placeholder.
    labels_placeholder: Labels placeholder.
  """
    # Note that the shapes of the placeholders match the shapes of the full
    # image and label tensors, except the first dimension is now batch_size
    # rather than the full size of the train or test data sets.
    images_placeholder = tf.placeholder(tf.float32, shape=(batch_size,
                                                           mnist.IMAGE_PIXELS), name="images")
    labels_placeholder = tf.placeholder(tf.int32, shape=(batch_size), name='labels')
    return images_placeholder, labels_placeholder


def fill_feed_dict(data_set, images_pl, labels_pl):
    """Fills the feed_dict for training the given step.

  A feed_dict takes the form of:
  feed_dict = {
      <placeholder>: <tensor of values to be passed for placeholder>,
      ....
  }

  Args:
    data_set: The set of images and labels, from input_data.read_data_sets()
    images_pl: The images placeholder, from placeholder_inputs().
    labels_pl: The labels placeholder, from placeholder_inputs().

  Returns:
    feed_dict: The feed dictionary mapping from placeholders to values.
  """
    # Create the feed_dict for the placeholders filled with the next
    # `batch size` examples.
    images_feed, labels_feed = data_set.next_batch(FLAGS.batch_size,
                                                   FLAGS.fake_data)
    feed_dict = {
        images_pl: images_feed,
        labels_pl: labels_feed,
    }
    return feed_dict


def do_eval(sess,
            eval_correct,
            images_placeholder,
            labels_placeholder,
            data_set):
    """Runs one evaluation against the full epoch of data.

  Args:
    sess: The session in which the model has been trained.
    eval_correct: The Tensor that returns the number of correct predictions.
    images_placeholder: The images placeholder.
    labels_placeholder: The labels placeholder.
    data_set: The set of images and labels to evaluate, from
      input_data.read_data_sets().
  """
    # And run one epoch of eval.
    true_count = 0  # Counts the number of correct predictions.
    steps_per_epoch = data_set.num_examples // FLAGS.batch_size
    num_examples = steps_per_epoch * FLAGS.batch_size
    for step in xrange(steps_per_epoch):
        feed_dict = fill_feed_dict(data_set,
                                   images_placeholder,
                                   labels_placeholder)
        true_count += sess.run(eval_correct, feed_dict=feed_dict)
    precision = float(true_count) / num_examples
    print('  Num examples: %d  Num correct: %d  Precision @ 1: %0.04f' %
          (num_examples, true_count, precision))


def step_summary(duration, feed_dict, sess, step, summary, summary_writer):
    # Print status to stdout.
    # print('Step %d: loss = %.2f (%.3f sec)' % (step, loss_value, duration))
    print('Step '+str(step))
    # Update the events file.
    summary_str = sess.run(summary, feed_dict=feed_dict)
    summary_writer.add_summary(summary_str, step)
    summary_writer.flush()


def checkpoint(data_sets, eval_correct, images_placeholder, labels_placeholder, saver, sess, step):
    checkpoint_file = os.path.join(FLAGS.log_dir, 'model.ckpt')
    saver.save(sess, checkpoint_file, global_step=step)
    # Evaluate against the training set.
    print('Training Data Eval:')
    do_eval(sess,
            eval_correct,
            images_placeholder,
            labels_placeholder,
            data_sets.train)
    # Evaluate against the validation set.
    print('Validation Data Eval:')
    do_eval(sess,
            eval_correct,
            images_placeholder,
            labels_placeholder,
            data_sets.validation)
    # Evaluate against the test set.
    print('Test Data Eval:')
    do_eval(sess,
            eval_correct,
            images_placeholder,
            labels_placeholder,
            data_sets.test)


def run_training():
    if not FLAGS.ipdb:
        import ipdb
        ipdb.set_trace = lambda: 0

    """Train MNIST for a number of steps."""
    # Get the sets of images and labels for training, validation, and
    # test on MNIST.
    data_sets = input_data.read_data_sets(FLAGS.input_data_dir, FLAGS.fake_data)
    # Tell TensorFlow that the model will be built into the default Graph.
    with tf.Graph().as_default():
        # Generate placeholders for the images and labels.
        images_placeholder, labels_placeholder = placeholder_inputs(
            FLAGS.batch_size)

        # Build a Graph that computes predictions from the inference model.
        simple_lissom = True
        # v1_layer, logits = supervised_gcal_graph.inference(images=images_placeholder,
        #                                                    image_shape=tf.TensorShape(
        #                                                        [mnist.IMAGE_SIZE, mnist.IMAGE_SIZE]),
        #                                                    simple_lissom=simple_lissom)

        v1, v1_layer = supervised_gcal_graph.inference_lissom(images_placeholder,
                                                           image_shape=tf.TensorShape(
                                                               [mnist.IMAGE_SIZE, mnist.IMAGE_SIZE]),
                                                           simple_lissom=simple_lissom)

        # Add to the Graph the Ops for loss calculation.
        # loss = supervised_gcal_graph.loss(logits=logits, labels=labels_placeholder)

        # Add to the Graph the Ops that calculate and apply gradients.
        # train_op_v1, train_op_classification = supervised_gcal_graph.training(v1_layer=v1_layer,
        #                                                                       loss=loss,
        #                                                                       learning_rate=FLAGS.learning_rate,
        #                                                                       simple_lissom=simple_lissom)

        train_op_v1 = supervised_gcal_graph.training_cortex(v1_layer=v1_layer,
                                                                              simple_lissom=simple_lissom)
        # # Add the Op to compare the logits to the labels during evaluation.
        # eval_correct = supervised_gcal_graph.evaluation(logits, labels_placeholder)

        # Build the summary Tensor based on the TF collection of Summaries.
        add_images_summaries(images_placeholder, simple_lissom, v1_layer)

        summary = tf.summary.merge_all()

        # Add the variable initializer Op.
        init = tf.global_variables_initializer()

        # Create a saver for writing training checkpoints.
        saver = tf.train.Saver()

        # Create a session for running Ops on the Graph.
        sess = tf.InteractiveSession()

        # Instantiate a SummaryWriter to output summaries and the Graph.
        summary_writer = tf.summary.FileWriter(FLAGS.log_dir, sess.graph)

        # And then after everything is built:

        # Run the Op to initialize the variables.
        sess.run(init)

        if FLAGS.debug:
            sess = tf_debug.LocalCLIDebugWrapperSession(sess, ui_type=FLAGS.ui_type)

            def has_all_ones(datum, tensor):
                return not np.all(tensor != np.ones(tensor.shape))

            sess.add_tensor_filter("has_all_ones", has_all_ones)

        # Start the training loop.
        for step in range(FLAGS.max_steps):
            start_time = time.time()

            # Fill a feed dictionary with the actual set of images and labels
            # for this particular training step.
            feed_dict = fill_feed_dict(data_sets.train,
                                       images_placeholder,
                                       labels_placeholder)

            # Run one step of the model.  The return values are the activations
            # from the `train_op` (which is discarded) and the `loss` Op.  To
            # inspect the values of your Ops or variables, you may include them
            # in the list passed to sess.run() and the value tensors will be
            # returned in the tuple from the call.
            # loss_value = sess.run([train_op_v2],
            # _, _, loss_value = sess.run([train_op_v1, train_op_classification, loss],
            #                             feed_dict=feed_dict)

            if step > 0:
                _ = sess.run([train_op_v1], feed_dict=feed_dict)

            duration = time.time() - start_time

            # Write the summaries and print an overview fairly often.
            if step < 10 or step % 100 == 0:
                # step_summary(duration, feed_dict, loss_value, sess, step, summary, summary_writer)
                step_summary(duration, feed_dict, sess, step, summary, summary_writer)

            # Save a checkpoint and evaluate the model periodically.
            # if (step + 1) % 1000 == 0 or (step + 1) == FLAGS.max_steps:
            #     checkpoint(data_sets, eval_correct, images_placeholder, labels_placeholder, saver, sess, step)


def main(_):
    if tf.gfile.Exists(FLAGS.log_dir):
        tf.gfile.DeleteRecursively(FLAGS.log_dir)
    tf.gfile.MakeDirs(FLAGS.log_dir)
    run_training()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--learning_rate',
        type=float,
        default=0.01,
        help='Initial learning rate.'
    )
    parser.add_argument(
        '--max_steps',
        type=int,
        default=101,
        help='Number of steps to run trainer.'
    )
    parser.add_argument(
        '--hidden1',
        type=int,
        default=128,
        help='Number of units in hidden layer 1.'
    )
    parser.add_argument(
        '--hidden2',
        type=int,
        default=32,
        help='Number of units in hidden layer 2.'
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=1,
        help='Batch size.  Must divide evenly into the dataset sizes.'
    )
    parser.add_argument(
        '--input_data_dir',
        type=str,
        default='/tmp/tensorflow/mnist.py/input_data',
        help='Directory to put the input data.'
    )
    parser.add_argument(
        '--log_dir',
        type=str,
        default='pipeline_logs',
        help='Directory to put the log data.'
    )
    parser.add_argument(
        '--fake_data',
        default=False,
        help='If true, uses fake data for unit testing.',
        action='store_true'
    )
    parser.add_argument(
        "--ipdb",
        default=False,
        help="Activate ipdb.set_trace()",
        action='store_true'
    )
    parser.add_argument(
        "--debug",
        default=False,
        help="Use debugger to track down bad values during training",
        action='store_true'
    )
    parser.add_argument(
        "--ui_type",
        type=str,
        default="readline",
        help="Command-line user interface type (curses | readline)"
    )

    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
