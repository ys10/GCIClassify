# coding=utf-8
import os
import tqdm
import argparse
import tensorflow as tf

from data_set import get_training_set, get_validation_set
from classify_model import ClassifyModel
from model_loader import load_model, save_model


def get_args():
    parser = argparse.ArgumentParser(description="WaveNet!")
    parser.add_argument("--save_path", type=str, default="./save/")
    parser.add_argument("--log_path", type=str, default="./log")
    parser.add_argument("--training_epochs", type=int, default=300)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--save_per_epochs", type=int, default=10)
    parser.add_argument("--validation_per_epochs", type=int, default=1)
    return parser.parse_args()


def main():
    args = get_args()
    net = ClassifyModel()
    graph = tf.Graph()
    with graph.as_default():
        with tf.variable_scope("data"):
            training_set, training_set_size = get_training_set(epochs=args.training_epochs, batch_size=args.batch_size)
            validation_set, validation_set_size = get_validation_set( epochs=args.training_epochs // args.validation_per_epochs)
            iterator = training_set.make_one_shot_iterator()
            next_element = iterator.get_next()
            training_init_op = iterator.make_initializer(training_set)
            validation_init_op = iterator.make_initializer(validation_set)

        with tf.variable_scope("classify_model"):
            tensor_dict = net.build(next_element)
            loss_summary = tf.summary.scalar("loss", tensor_dict["loss"])
            accuracy_summary = tf.summary.scalar("accuracy", tensor_dict["accuracy"])
            recall_summary = tf.summary.scalar("recall", tensor_dict["recall"])
            precision_summary = tf.summary.scalar("precision", tensor_dict["precision"])
            global_step = tf.Variable(0, dtype=tf.int32, name="global_step")
            opt = tf.train.AdamOptimizer(1e-3)
            upd = opt.minimize(tensor_dict["loss"], global_step=global_step)
            saver = tf.train.Saver()

    tf.logging.set_verbosity(tf.logging.INFO)
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.Session(graph=graph, config=config) as sess:
        save_path = os.path.join(args.save_path, net.name)
        if not load_model(saver, sess, save_path):
            sess.run([tf.global_variables_initializer(), tf.local_variables_initializer()])
        training_writer = tf.summary.FileWriter(args.log_path + "/training", sess.graph)
        validation_writer = tf.summary.FileWriter(args.log_path + "/validation", sess.graph)
        global_step_eval = sess.run(global_step)
        training_steps = args.training_epochs * training_set_size // args.batch_size
        save_steps = args.save_per_epochs * training_set_size // args.batch_size
        validation_steps = args.validation_per_epochs * training_set_size // args.batch_size
        pbar = tqdm.tqdm(total=training_steps)
        pbar.update(global_step_eval)
        sess.run(training_init_op)
        while global_step_eval < training_steps:
            training_list = [loss_summary, accuracy_summary, recall_summary, precision_summary, global_step, upd]
            training_loss_summary_eval, training_accuracy_summary_eval, training_recall_summary_eval, training_precision_summary_eval, global_step_eval, _ = sess.run(training_list)
            training_writer.add_summary(training_loss_summary_eval, global_step=global_step_eval)
            training_writer.add_summary(training_accuracy_summary_eval, global_step=global_step_eval)
            training_writer.add_summary(training_recall_summary_eval, global_step=global_step_eval)
            training_writer.add_summary(training_precision_summary_eval, global_step=global_step_eval)
            """save model"""
            if global_step_eval % save_steps == 0:
                if not os.path.exists(args.save_path) or not os.path.isdir(args.save_path):
                    os.makedirs(args.save_path)
                save_model(saver, sess, save_path, global_step_eval)
            """validation"""
            if global_step_eval % validation_steps == 0:
                sess.run(validation_init_op)
                validation_list = [loss_summary, accuracy_summary, recall_summary,
                                 precision_summary]
                validation_loss_summary_eval, validation_accuracy_summary_eval, validation_recall_summary_eval, validation_precision_summary_eval = sess.run(
                    validation_list)
                validation_writer.add_summary(validation_loss_summary_eval, global_step=global_step_eval)
                validation_writer.add_summary(validation_accuracy_summary_eval, global_step=global_step_eval)
                validation_writer.add_summary(validation_recall_summary_eval, global_step=global_step_eval)
                validation_writer.add_summary(validation_precision_summary_eval, global_step=global_step_eval)
                tf.logging.info("Validation done.")
                sess.run(training_init_op)
            pbar.update(1)

    tf.logging.info("Congratulations!")


if __name__ == "__main__":
    main()
