# coding=utf-8
import argparse


def get_rab_set_args():
    parser = argparse.ArgumentParser(description="rab_set")
    parser.add_argument("--training_set_name", type=str, default="rab_training")
    parser.add_argument("--validation_set_name", type=str, default="rab_validation")
    parser.add_argument("--training_set_size", type=int, default=308514)
    parser.add_argument("--validation_set_size", type=int, default=16750)
    return parser.parse_args()


def get_ked_set_args():
    parser = argparse.ArgumentParser(description="rab_set")
    parser.add_argument("--training_set_name", type=str, default="ked_training")
    parser.add_argument("--validation_set_name", type=str, default="ked_validation")
    parser.add_argument("--training_set_size", type=int, default=220039)
    parser.add_argument("--validation_set_size", type=int, default=27494)
    return parser.parse_args()


def get_bdl_set_args():
    parser = argparse.ArgumentParser(description="bdl_set")
    parser.add_argument("--training_set_name", type=str, default="bdl_training")
    parser.add_argument("--validation_set_name", type=str, default="bdl_validation")
    parser.add_argument("--training_set_size", type=int, default=710776)
    parser.add_argument("--validation_set_size", type=int, default=68066)
    return parser.parse_args()


def get_jmk_set_args():
    parser = argparse.ArgumentParser(description="jmk_set")
    parser.add_argument("--training_set_name", type=str, default="jmk_training")
    parser.add_argument("--validation_set_name", type=str, default="jmk_validation")
    parser.add_argument("--training_set_size", type=int, default=360874)
    parser.add_argument("--validation_set_size", type=int, default=37100)
    return parser.parse_args()


def get_slt_set_args():
    parser = argparse.ArgumentParser(description="slt_set")
    parser.add_argument("--training_set_name", type=str, default="slt_training")
    parser.add_argument("--validation_set_name", type=str, default="slt_validation")
    parser.add_argument("--training_set_size", type=int, default=668007)
    parser.add_argument("--validation_set_size", type=int, default=67269)
    return parser.parse_args()


def get_mix2_set_args():
    parser = argparse.ArgumentParser(description="mix2_set")
    parser.add_argument("--training_set_name", type=str, default="mix2_training")
    parser.add_argument("--validation_set_name", type=str, default="mix2_validation")
    parser.add_argument("--training_set_size", type=int, default=27114)
    parser.add_argument("--validation_set_size", type=int, default=25788)
    return parser.parse_args()


def get_mix3_set_args():
    parser = argparse.ArgumentParser(description="mix3_set")
    parser.add_argument("--training_set_name", type=str, default="mix3_training")
    parser.add_argument("--validation_set_name", type=str, default="mix3_validation")
    parser.add_argument("--training_set_size", type=int, default=243455)
    parser.add_argument("--validation_set_size", type=int, default=113693)
    return parser.parse_args()


def get_mix4_set_args():
    parser = argparse.ArgumentParser(description="mix4_set")
    parser.add_argument("--training_set_name", type=str, default="mix4_training")
    parser.add_argument("--validation_set_name", type=str, default="mix4_validation")
    parser.add_argument("--training_set_size", type=int, default=1176559)
    parser.add_argument("--validation_set_size", type=int, default=64120)
    return parser.parse_args()
