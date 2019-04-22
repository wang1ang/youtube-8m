import math

import models
import video_level_models
import tensorflow as tf
import model_utils as utils
import tensorflow.contrib.slim as slim
from tensorflow import flags

FLAGS = flags.FLAGS
#flags.DEFINE_integer("iterations", 30, "Number of frames per batch for DBoF.")
#flags.DEFINE_bool("sample_random_frames", True, "If true samples random frames (for frame level models). If false, a random sequence of frames is sampled instead.")
#flags.DEFINE_string("video_level_classifier_model", "MoeModel", "Some Frame-Level models can be decomposed into a generalized pooling operation followed by a classifier layer")
# for NeXtVLAD
flags.DEFINE_integer("nextvlad_cluster_size", 128, "Number of units in the NeXtVLAD cluster layer.")
flags.DEFINE_integer("nextvlad_hidden_size", 2048, "Number of units in the NeXtVLAD hidden layer.")
flags.DEFINE_integer("groups", 8, "number of groups in VLAD encoding")
flags.DEFINE_float("drop_rate", 0.5, "dropout ratio after VLAD encoding")
flags.DEFINE_integer("expansion", 2, "expansion ratio in Group NetVlad")
flags.DEFINE_integer("gating_reduction", 8, "reduction factor in se context gating")



class NeXtVLADModel(models.BaseModel):
    def create_model(self,
                     model_input,
                     vocab_size,
                     num_frames,
                     iterations=None,
                     add_batch_norm=None,
                     sample_random_frames=None,
                     cluster_size=None,
                     hidden_size=None,

                     is_training=True,
                     expansion=2, 
                     groups=None,
                     #mask=None,
                     drop_rate=0.5,
                     gating_reduction=None,
                     **unused_params):
        iterations = iterations or FLAGS.iterations
        add_batch_norm = add_batch_norm or FLAGS.dbof_add_batch_norm
        random_frames = FLAGS.sample_random_frames if sample_random_frames is None else sample_random_frames
        cluster_size = cluster_size or FLAGS.nextvlad_cluster_size
        hidden_size = hidden_size or FLAGS.nextvlad_hidden_size
        groups = groups or FLAGS.groups
        gating_reduction = gating_reduction or FLAGS.gating_reduction

        num_frames_exp = tf.cast(tf.expand_dims(num_frames, 1), tf.float32)
        if random_frames:
            model_input = utils.SampleRandomFrames(model_input, num_frames_exp, iterations)
        else:
            model_input = utils.SampleRandomSequence(model_input, num_frames_exp, iterations)
        max_frames = model_input.get_shape().as_list()[1]
        feature_size = model_input.get_shape().as_list()[2]
        #reshaped_input = tf.reshape(model_input, [-1, feature_size])
        #tf.summary.histogram("input_hist", reshaped_input)
        
        mask = tf.sequence_mask(num_frames, max_frames, dtype=tf.float32)

        input = slim.fully_connected(model_input, expansion * feature_size, activation_fn=None,
                                     weights_initializer=slim.variance_scaling_initializer())

        attention = slim.fully_connected(model_input, groups, activation_fn=tf.nn.sigmoid,
                                         weights_initializer=slim.variance_scaling_initializer())

        if mask is not None:
            attention = tf.multiply(attention, tf.expand_dims(mask, -1))
        attention = tf.reshape(attention, [-1, max_frames * groups, 1])
        tf.summary.histogram("sigmoid_attention", attention)
        reduce_size = expansion * feature_size // groups

        cluster_weights = tf.get_variable("cluster_weights",
                                          [expansion*feature_size, groups*cluster_size],
                                          initializer=slim.variance_scaling_initializer()
                                          )

        # tf.summary.histogram("cluster_weights", cluster_weights)
        reshaped_input = tf.reshape(input, [-1, expansion * feature_size])
        activation = tf.matmul(reshaped_input, cluster_weights)

        activation = slim.batch_norm(
            activation,
            center=True,
            scale=True,
            is_training=is_training,
            scope="cluster_bn",
            fused=False)

        activation = tf.reshape(activation, [-1, max_frames * groups, cluster_size])
        activation = tf.nn.softmax(activation, axis=-1)
        activation = tf.multiply(activation, attention)
        # tf.summary.histogram("cluster_output", activation)
        a_sum = tf.reduce_sum(activation, -2, keep_dims=True)

        cluster_weights2 = tf.get_variable("cluster_weights2",
                                           [1, reduce_size, cluster_size],
                                           initializer=slim.variance_scaling_initializer()
                                           )
        a = tf.multiply(a_sum, cluster_weights2)

        activation = tf.transpose(activation, perm=[0, 2, 1])

        reshaped_input = tf.reshape(input, [-1, max_frames * groups, reduce_size])
        vlad = tf.matmul(activation, reshaped_input)
        vlad = tf.transpose(vlad, perm=[0, 2, 1])
        vlad = tf.subtract(vlad, a)

        vlad = tf.nn.l2_normalize(vlad, 1)

        vlad = tf.reshape(vlad, [-1, cluster_size * reduce_size])
        vlad = slim.batch_norm(vlad,
                center=True,
                scale=True,
                is_training=is_training,
                scope="vlad_bn",
                fused=False)





        if drop_rate > 0.:
            vlad = slim.dropout(vlad, keep_prob=1. - drop_rate, is_training=is_training, scope="vlad_dropout")

        vlad_dim = vlad.get_shape().as_list()[1]
        print("VLAD dimension", vlad_dim)
        hidden_weights = tf.get_variable("hidden_weights",
                                          [vlad_dim, hidden_size],
                                          initializer=slim.variance_scaling_initializer())

        activation = tf.matmul(vlad, hidden_weights)
        activation = slim.batch_norm(
            activation,
            center=True,
            scale=True,
            is_training=is_training,
            scope="hidden_bn",
            fused=False)

        activation = tf.nn.relu(activation, name='embedding1')

        gating_weights_1 = tf.get_variable("gating_weights_1",
                                           [hidden_size, hidden_size // gating_reduction],
                                           initializer=slim.variance_scaling_initializer())

        gates = tf.matmul(activation, gating_weights_1)

        gates = slim.batch_norm(
            gates,
            center=True,
            scale=True,
            is_training=is_training,
            activation_fn=slim.nn.relu,
            scope="gating_bn")

        gating_weights_2 = tf.get_variable("gating_weights_2",
                                           [hidden_size // gating_reduction, hidden_size],
                                           initializer=slim.variance_scaling_initializer()
                                           )
        gates = tf.matmul(gates, gating_weights_2)

        gates = tf.sigmoid(gates)
        tf.summary.histogram("final_gates", gates)

        activation = tf.multiply(activation, gates, name="embedding2")

        aggregated_model = getattr(video_level_models,
                                   FLAGS.video_level_classifier_model)

        return aggregated_model().create_model(
            model_input=activation,
            vocab_size=vocab_size,
            is_training=is_training,
            **unused_params)