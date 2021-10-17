
import argparse
import tensorflow as tf
from datasets.twitter.dataset_loader import TwitterLoader
from datasets.weibo.dataset_loader import WeiboLoader
from models.inception import Inception
from models.parallel_conv import ParallelConv
from models.resnet import Resnet
from models.sequence_conv import SequenceConv
from models.vgg import VGG




if __name__ == '__main__':
    import os
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    config = tf.compat.v1.ConfigProto(allow_soft_placement=True, log_device_placement=True)
    config.gpu_options.allow_growth = True
    tf.compat.v1.Session(config=config)

    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, required=True)
    parser.add_argument('--model', type=str, required=True)
    parser.add_argument('--use_optuna', type=int, required=False)
    parser.add_argument('--extra', type=str, required=False)

    args = parser.parse_args()

    dataset = None
    if args.data == 'twitter':
        dataset = TwitterLoader()
    elif args.data == 'weibo':
        dataset = WeiboLoader()
    else:
        print(dataset)
        raise Exception('Invalid local_datasets name!')

    model = None
    if args.model == 'vgg':
        model = VGG(dataset)
    elif args.model == 'resnet':
        model = Resnet(dataset)
    elif args.model == 'inception':
        model = Inception(dataset)
    elif args.model == 'p_conv':
        model = ParallelConv(dataset)
    elif args.model == 's_conv':
        model = SequenceConv(dataset)
    else:
        print(model)
        raise Exception('Invalid model name!')

    if args.use_optuna:
        model.optuna_main(args.use_optuna)
        model.main()
    else:
        model.main()

