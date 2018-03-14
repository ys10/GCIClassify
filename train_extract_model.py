# coding=utf-8
import argparse
import os

import tensorflow as tf
import tqdm

from feature_extraction.extract_model import ExtractModel
from feature_extraction.data_set import get_testing_set
from model_loader import load_model, save_model
from data_set_args import get_rab_set_args, get_ked_set_args,\
    get_bdl_set_args, get_jmk_set_args, get_slt_set_args,\
    get_mix2_set_args, get_mix3_set_args


def get_args():
    parser = argparse.ArgumentParser(description="GlottalNet")
    parser.add_argument("--save_path", type=str, default="./save/mix3/")
    parser.add_argument("--log_path", type=str, default="./log/mix3/")
    parser.add_argument("--training_epochs", type=int, default=100)
    parser.add_argument("--training_batch_size", type=int, default=128)
    parser.add_argument("--validation_batch_size", type=int, default=128)
    parser.add_argument("--save_per_epochs", type=int, default=10)
    parser.add_argument("--validation_per_epochs", type=int, default=1)
    return parser.parse_args()


def main():
    tf.logging.set_verbosity(tf.logging.INFO)
    args = get_args()
    data_set_args = get_mix3_set_args()
    net = ExtractModel()
    graph = tf.Graph()
    with graph.as_default():
        with tf.variable_scope("data"):
            training_set = get_testing_set(key=data_set_args.training_set_name,
                                           epochs=args.training_epochs, batch_size=args.training_batch_size)
            validation_set = get_testing_set(key=data_set_args.validation_set_name,
                                             epochs=args.training_epochs // args.validation_per_epochs,
                                             batch_size=args.validation_batch_size)
            iterator = training_set.make_one_shot_iterator()
            next_element = iterator.get_next()
            training_init_op = iterator.make_initializer(training_set)
            validation_init_op = iterator.make_initializer(validation_set)

        with tf.variable_scope("extract_model"):
            tensor_dict = net.build(next_element, training=True)
            """training summary"""
            loss_summary = tf.summary.scalar("loss", tensor_dict["loss"])
            accuracy_summary = tf.summary.scalar("accuracy", tensor_dict["accuracy"])
            recall_summary = tf.summary.scalar("recall", tensor_dict["recall"])
            precision_summary = tf.summary.scalar("precision", tensor_dict["precision"])
            f1_score_summary = tf.summary.scalar("f1_score", tensor_dict["f1_score"])
            """validation summary"""
            validation_loss = tf.placeholder(tf.float32, shape=())
            validation_accuracy = tf.placeholder(tf.float32, shape=())
            validation_recall = tf.placeholder(tf.float32, shape=())
            validation_precision = tf.placeholder(tf.float32, shape=())
            validation_f1_score = tf.placeholder(tf.float32, shape=())
            validation_loss_summary = tf.summary.scalar("loss", validation_loss)
            validation_accuracy_summary = tf.summary.scalar("accuracy", validation_accuracy)
            validation_recall_summary = tf.summary.scalar("recall", validation_recall)
            validation_precision_summary = tf.summary.scalar("precision", validation_precision)
            validation_f1_score_summary = tf.summary.scalar("f1_score", validation_f1_score)
            """training"""
            global_step = tf.Variable(0, dtype=tf.int32, name="global_step")
            opt = tf.train.AdamOptimizer(1e-3)
            extra_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(extra_update_ops):
                upd = opt.minimize(tensor_dict["loss"], global_step=global_step)
            saver = tf.train.Saver(max_to_keep=50)

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.Session(graph=graph, config=config) as sess:
        sess.run([tf.global_variables_initializer(), tf.local_variables_initializer()])
        save_path = os.path.join(args.save_path, net.name)
        if not load_model(saver, sess, save_path):
            tf.logging.info("Run on an initialized graph.")
            sess.run([tf.global_variables_initializer(), tf.local_variables_initializer()])
        training_writer = tf.summary.FileWriter(os.path.join(args.log_path, "training"), sess.graph)
        validation_writer = tf.summary.FileWriter(os.path.join(args.log_path, "validation"), sess.graph)
        global_step_eval = sess.run(global_step)
        training_steps = args.training_epochs * data_set_args.training_set_size // args.training_batch_size
        save_steps = args.save_per_epochs * data_set_args.training_set_size // args.training_batch_size
        validation_steps = args.validation_per_epochs * data_set_args.training_set_size // args.training_batch_size
        pbar = tqdm.tqdm(total=training_steps)
        pbar.update(global_step_eval)
        sess.run(training_init_op)
        while global_step_eval < training_steps:
            """validation"""
            if global_step_eval % validation_steps == 0:
                sess.run(validation_init_op)
                total_loss = 0.0
                total_accuracy = 0.0
                total_recall = 0.0
                total_precision = 0.0
                validation_steps = data_set_args.validation_set_size // args.validation_batch_size
                for s in range(validation_steps):
                    tensor_dict_eval = sess.run(tensor_dict)
                    total_loss += tensor_dict_eval["loss"]
                    total_accuracy += tensor_dict_eval["accuracy"]
                    total_recall += tensor_dict_eval["recall"]
                    total_precision += tensor_dict_eval["precision"]
                total_loss /= validation_steps
                total_accuracy /= validation_steps
                total_recall /= validation_steps
                total_precision /= validation_steps
                total_f1_score = 2 * total_recall * total_precision / (total_recall + total_precision)
                feed_dict = {validation_loss: total_loss, validation_accuracy: total_accuracy, validation_recall: total_recall,
                                   validation_precision: total_precision, validation_f1_score: total_f1_score}
                validation_list = [validation_loss_summary, validation_accuracy_summary, validation_recall_summary,
                                   validation_precision_summary, validation_f1_score_summary]
                validation_loss_summary_eval, validation_accuracy_summary_eval, validation_recall_summary_eval,\
                    validation_precision_summary_eval, validation_f1_score_summary_eval = sess.run(validation_list,
                                                                                                   feed_dict=feed_dict)
                validation_writer.add_summary(validation_loss_summary_eval, global_step=global_step_eval)
                validation_writer.add_summary(validation_accuracy_summary_eval, global_step=global_step_eval)
                validation_writer.add_summary(validation_recall_summary_eval, global_step=global_step_eval)
                validation_writer.add_summary(validation_precision_summary_eval, global_step=global_step_eval)
                validation_writer.add_summary(validation_f1_score_summary_eval, global_step=global_step_eval)
                tf.logging.info("Validation done.")
                sess.run(training_init_op)
            """training"""
            training_list = [loss_summary, accuracy_summary, recall_summary, precision_summary,
                             f1_score_summary, global_step, upd]
            training_loss_summary_eval, training_accuracy_summary_eval, training_recall_summary_eval,\
                training_precision_summary_eval, training_f1_score_summary_eval, global_step_eval,\
                _ = sess.run(training_list)
            training_writer.add_summary(training_loss_summary_eval, global_step=global_step_eval)
            training_writer.add_summary(training_accuracy_summary_eval, global_step=global_step_eval)
            training_writer.add_summary(training_recall_summary_eval, global_step=global_step_eval)
            training_writer.add_summary(training_precision_summary_eval, global_step=global_step_eval)
            training_writer.add_summary(training_f1_score_summary_eval, global_step=global_step_eval)
            """save model"""
            if global_step_eval % save_steps == 0:
                if not os.path.exists(args.save_path) or not os.path.isdir(args.save_path):
                    os.makedirs(args.save_path)
                save_model(saver, sess, save_path, global_step_eval)
            pbar.update(1)

    tf.logging.info("Congratulations!")


if __name__ == "__main__":
    main()
