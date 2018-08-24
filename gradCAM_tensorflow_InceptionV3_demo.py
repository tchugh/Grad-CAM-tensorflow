# Replace vanila relu to guided relu to get guided backpropagation.
import tensorflow as tf

from tensorflow.python.framework import ops
from tensorflow.python.ops import gen_nn_ops
import numpy as np
import utils
import os


slim = tf.contrib.slim
os.environ['CUDA_VISIBLE_DEVICES']='3'
from slim.nets import inception_v4

@ops.RegisterGradient("GuidedRelu")
def _GuidedReluGrad(op, grad):
    return tf.where(0. < grad, gen_nn_ops.relu_grad(grad, op.outputs[0]), tf.zeros(grad.get_shape()))

# Create mini-batch for demo

img1 = utils.load_image("./demo.png", normalize=False, size=299)
img2 = utils.load_image("./shihtzu_mypuppy.jpg", normalize=False, size=299)
img3 = utils.load_image("./tiger.jpg", normalize=False, size=299)

batch1_img = img1.reshape((1, 299, 299, 3))
batch1_label = np.array([1 if i == 242 else 0 for i in range(1000)])  # 1-hot result for Boxer
batch1_label = batch1_label.reshape(1, -1)

batch2_img = img2.reshape((1, 299, 299, 3))
batch2_label = np.array([1 if i == 155 else 0 for i in range(1000)])  # 1-hot result for Shih-Tzu
batch2_label = batch2_label.reshape(1, -1)

batch3_img = img3.reshape((1, 299, 299, 3))
batch3_label = np.array([1 if i == 292 else 0 for i in range(1000)])  # 1-hot result for tiger
batch3_label = batch3_label.reshape(1, -1)

batch_img = np.concatenate((batch1_img, batch2_img, batch3_img), 0)
batch_label = np.concatenate((batch1_label, batch2_label, batch3_label), 0)
batch_size = 3

# batch_img = np.concatenate((batch1_img), 0)
# batch_label = np.concatenate((batch1_label), 0)
# batch_size = 1
# batch_label = batch_label.reshape(batch_size, -1)

# Create tensorflow graph for evaluation
eval_graph = tf.Graph()
with eval_graph.as_default():
    with eval_graph.gradient_override_map({'Relu': 'GuidedRelu'}):
        images = tf.placeholder("float", [batch_size, 299, 299, 3])
        labels = tf.placeholder(tf.float32, [batch_size, 1000])

        preprocessed_images = utils.resnet_preprocess(images)

        with slim.arg_scope(inception_v4.inception_v4_arg_scope()):
            #with slim.arg_scope():
                # is_training=False means batch-norm is not in training mode. Fixing batch norm layer.
                # net is logit for resnet_v1. See is_training messing up issue: https://github.com/tensorflow/tensorflow/issues/4887
            net, end_points = inception_v4.inception_v4(preprocessed_images, 1000, is_training=False)

        prob = end_points['Predictions']  # after softmax

        cost = (-1) * tf.reduce_sum(tf.multiply(labels, tf.log(prob)), axis=1)
        print('cost:', cost)
        y_c = tf.reduce_sum(tf.multiply(net, labels), axis=1)
        print('y_c:', y_c)

        # Get last convolutional layer gradient for generating gradCAM visualization
        print('endpoints:', end_points.keys())
        target_conv_layer = end_points['Mixed_7c']
        target_conv_layer_grad = tf.gradients(y_c, target_conv_layer)[0]

        # Guided backpropagtion back to input layer
        gb_grad = tf.gradients(cost, images)[0]

        init = tf.global_variables_initializer()

        # Load resnet v1 weights

        # latest_checkpoint = tf.train.latest_checkpoint("model/resnet_v1_50.ckpt")
        latest_checkpoint = "model/inception_v4.ckpt"
        ## Optimistic restore.
        reader = tf.train.NewCheckpointReader(latest_checkpoint)
        saved_shapes = reader.get_variable_to_shape_map()
        variables_to_restore = tf.global_variables()
        for var in variables_to_restore:
            if not var.name.split(':')[0] in saved_shapes:
                print("WARNING. Saved weight not exists in checkpoint. Init var:", var.name)
            else:
                print("Load saved weight:", var.name)
                pass

        var_names = sorted([(var.name, var.name.split(':')[0]) for var in variables_to_restore
                            if var.name.split(':')[0] in saved_shapes])
        restore_vars = []
        with tf.variable_scope('', reuse=True):
            for var_name, saved_var_name in var_names:
                try:
                    curr_var = tf.get_variable(saved_var_name)
                    var_shape = curr_var.get_shape().as_list()
                    if var_shape == saved_shapes[saved_var_name]:
                        # print("restore var:", saved_var_name)
                        restore_vars.append(curr_var)
                except ValueError:
                    print("Ignore due to ValueError on getting var:", saved_var_name)
        saver = tf.train.Saver(restore_vars)

# Run tensorflow

with tf.Session(graph=eval_graph) as sess:
    sess.run(init)
    # sess.run(tf.local_variables_initializer())
    saver.restore(sess, latest_checkpoint)

    prob = sess.run(prob, feed_dict={images: batch_img})

    # gb_grad_value, target_conv_layer_value, target_conv_layer_grad_value = sess.run([gb_grad, target_conv_layer, target_conv_layer_grad], feed_dict={images: batch_img, labels: prob})
    gb_grad_value, target_conv_layer_value, target_conv_layer_grad_value = sess.run(
        [gb_grad, target_conv_layer, target_conv_layer_grad], feed_dict={images: batch_img, labels: batch_label})

    for i in range(batch_size):
        # print('See visualization of below category')
        # utils.print_prob(batch_label[i], './synset.txt')
        utils.print_prob(prob[i], './synset.txt')
        # print('gb_grad_value[i]:', gb_grad_value[i])
        # print('gb_grad_value[i] shape:', gb_grad_value[i].shape)
        utils.visualize(batch_img[i], target_conv_layer_value[i], target_conv_layer_grad_value[i], gb_grad_value[i], size=299)