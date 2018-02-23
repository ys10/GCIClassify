# coding=utf-8
import os
import tqdm
import argparse
import tensorflow as tf

from data_set import get_testing_set
from classify_model import ClassifyModel
from model_loader import load_model


def get_args():
    parser = argparse.ArgumentParser(description="DiscriminateNetwork")
    parser.add_argument("--save_path", type=str, default="./save/")
    parser.add_argument("--testing_set_name", type=str, default="cmu/ked")
    parser.add_argument("--log_path", type=str, default="./log/")
    parser.add_argument("--testing_epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=64)
    return parser.parse_args()


def main():
    tf.logging.set_verbosity(tf.logging.INFO)
    args = get_args()
    tf.logging.info("Test model on set: " + args.testing_set_name)
    net = ClassifyModel()
    graph = tf.Graph()
    with graph.as_default():
        with tf.variable_scope("data"):
            testing_set, testing_set_size = get_testing_set(key=args.testing_set_name,
                                                            epochs=args.testing_epochs, batch_size=args.batch_size)
            iterator = testing_set.make_one_shot_iterator()
            next_element = iterator.get_next()
            testing_init_op = iterator.make_initializer(testing_set)

        with tf.variable_scope("classify_model"):
            tensor_dict = net.build(next_element, training=False)
            loss = tensor_dict["loss"]
            accuracy = tensor_dict["accuracy"]
            recall = tensor_dict["recall"]
            precision = tensor_dict["precision"]
            saver = tf.train.Saver()

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.Session(graph=graph, config=config) as sess:
        sess.run([tf.global_variables_initializer(), tf.local_variables_initializer()])
        save_path = os.path.join(args.save_path, net.name)
        if not load_model(saver, sess, save_path):
            sess.run([tf.global_variables_initializer(), tf.local_variables_initializer()])
        total_loss = 0.0
        total_accuracy = 0.0
        total_recall = 0.0
        total_precision = 0.0
        global_step_eval = 0
        testing_steps = args.testing_epochs * testing_set_size // args.batch_size
        pbar = tqdm.tqdm(total=testing_steps)
        pbar.update(global_step_eval)
        sess.run(testing_init_op)
        for global_step_eval in range(testing_steps):
            testing_list = [loss, accuracy, recall, precision]
            loss_eval, accuracy_eval, recall_eval, precision_eval = sess.run(testing_list)
            total_loss += loss_eval
            total_accuracy += accuracy_eval
            total_recall += recall_eval
            total_precision += precision_eval
            pbar.update(1)
        total_loss /= global_step_eval
        total_accuracy /= global_step_eval
        total_recall /= global_step_eval
        total_precision /= global_step_eval
        total_f1_score = 2 * total_recall * total_precision / (total_recall + total_precision)
        tf.logging.info("Average loss: {:.6f}, accuracy: {:.6f}, recall: {:.6f}, precision: {:.6f}, f1_score: {:.6f}"
                        .format(total_loss, total_accuracy, total_recall, total_precision, total_f1_score))
    tf.logging.info("Congratulations!")


if __name__ == "__main__":
    main()
