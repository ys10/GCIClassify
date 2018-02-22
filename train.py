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
    parser.add_argument("--training_steps", type=int, default=25000)
    parser.add_argument("--validation_size", type=int, default=152)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--save_per_steps", type=int, default=5000)
    parser.add_argument("--validation_per_steps", type=int, default=1000)
    return parser.parse_args()


def main():
    args = get_args()
    net = ClassifyModel()
    graph = tf.Graph()
    with graph.as_default():
        with tf.variable_scope("training_data"):
            training_set = get_training_set()
            training_iterator = training_set.make_one_shot_iterator()
            training_element = training_iterator.get_next()

        with tf.variable_scope("validation_data"):
            validation_set = get_validation_set()
            validation_iterator = validation_set.make_one_shot_iterator()
            validation_element = validation_iterator.get_next()

        with tf.variable_scope("classify_model"):
            training_tensor_dict = net.build(training_element)
            training_loss_summary = tf.summary.scalar("training_loss", training_tensor_dict["loss"])
            global_step = tf.Variable(0, dtype=tf.int32, name="global_step")
            opt = tf.train.AdamOptimizer(1e-3)
            upd = opt.minimize(training_tensor_dict["loss"], global_step=global_step)
            saver = tf.train.Saver()

    tf.logging.set_verbosity(tf.logging.INFO)
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.Session(graph=graph, config=config) as sess:
        save_path = os.path.join(args.save_path, net.name)
        if not load_model(saver, sess, save_path):
            sess.run([tf.global_variables_initializer(), tf.local_variables_initializer()])
        training_writer = tf.summary.FileWriter(args.log_path + "/training", sess.graph)
        global_step_eval = sess.run(global_step)
        pbar = tqdm.tqdm(total=args.training_steps)
        pbar.update(global_step_eval)
        while global_step_eval < args.training_steps:
            training_loss_summary_eval, global_step_eval, _ = sess.run([training_loss_summary, global_step, upd])
            training_writer.add_summary(training_loss_summary_eval, global_step=global_step_eval)
            """save model"""
            if global_step_eval % args.save_per_steps == 0:
                if not os.path.exists(args.save_path) or not os.path.isdir(args.save_path):
                    os.makedirs(args.save_path)
                save_model(saver, sess, save_path, global_step_eval)
            pbar.update(1)

    tf.logging.info("Congratulations!")


if __name__ == "__main__":
    main()
