from src.finetuning import finetune
import argparse
import os
import yaml


def _parse_args(verbose=True):
    parser = argparse.ArgumentParser(description='')

    parser.add_argument('--conf',
                        type=str,
                        default='configs/ft_cifar10_config.yaml',
                        help='Pass Config file path')

    parser.add_argument('--log_path',
                        type=str,
                        default='./default_log',
                        help='Pass log file path')

    parser.add_argument('--op_model_path',
                        type=str,
                        default='./default_model',
                        help='model to save - Full Path')

    parser.add_argument('--ip_model_path',
                        type=str,
                        default=None,
                        help='starting model path - if None then rand init')

    parser.add_argument('--train_mode',
                        type=str,
                        default='finetune',
                        help='encode/finetune/linear_probe/knn_eval')

    parser.add_argument('--port',
                        type=str,
                        default='12345',
                        help='Specify PORT / for DDP')

    parser.add_argument('--n_repeat',
                        type=int,
                        default=1,
                        help='Specify number of repeat runs')

    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument("--local_world_size", type=int, default=1)

    args = parser.parse_args()
    verbose and print(args)

    return args


def run_main():
    """
    Driver Script
    """
    # Parse Arguments
    args = _parse_args()
    if args.train_mode not in ['encode', 'finetune', 'linear_probe', 'knn_eval']:
        raise ValueError
    config = yaml.load(open(args.conf), Loader=yaml.FullLoader)

    # Prepare for Logging
    if not os.path.exists(os.path.split(args.log_path)[0]):
        os.makedirs(os.path.split(args.log_path)[0])
    if not os.path.exists(os.path.split(args.op_model_path)[0]):
        os.makedirs(os.path.split(args.op_model_path)[0])

    if args.train_mode == 'finetune':
        finetune(args=args, config=config)
    else:
        raise NotImplementedError


if __name__ == '__main__':
    run_main()
