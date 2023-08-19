from module import get_pretrain_model,pre_train
from dataset import get_data
import tensorflow as tf
import argparse

parser = argparse.ArgumentParser(description='training setup')

parser.add_argument('--dir', type=str, default='datasets/sub', help='dataset path')
parser.add_argument('--dataset', type=str, default='ucihar', choices=['ucihar', 'motion', 'uschad'], help='dataset')
parser.add_argument('--type', type=str, default='channel', choices=['time','spantime','spantime_channel','time_channel'\
    ,'channel'], help='masking strategies')
parser.add_argument('--channel_mask', type=int, default=3, help='number of channel masks')
parser.add_argument('--time_mask', type=int, default=3, help='time mask ratio')
parser.add_argument('--alpha', type=int, default=0.5, help='the hyperparameter alpha')

# encoder
parser.add_argument('--num_layers', type=int, default=3, help='number of attention blocks')
parser.add_argument('--num_heads', type=int, default=4, help='the number of heads in multi-head attention')
parser.add_argument('--dff', type=int, default=256, help='dff')
parser.add_argument('--d_model', type=int, default=128, help='Transformer Encoder Embedding Layer Size')

# train
parser.add_argument('--batch_size', type=int, default=256, help='batch size of training')
parser.add_argument('--epoch', type=int, default=150, help='number of training epochs')
parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
parser.add_argument('--seed', type=int, default=100, help='random seed for datasets division')

if __name__ == '__main__':
    args = parser.parse_args()

    x_train, y_train, _, _, _, _ = get_data(args.dir, args.dataset, transformer=True, divide_seed=args.seed)
    n_timesteps, n_features, n_outputs = x_train.shape[1], x_train.shape[2], y_train.shape[1]

    pretrain_model = get_pretrain_model(args.num_layers, args.d_model, args.num_heads, args.dff, \
                                        maximum_position_encoding=n_timesteps, n_features = n_features)

    optimizer = tf.keras.optimizers.Adam(args.lr)
    loss_func = tf.keras.losses.MeanSquaredError()
    pre_train(pretrain_model, args.dataset, x_train, args.epoch, args.batch_size, optimizer, loss_func, args.type, \
              n_timesteps, args.time_mask, n_features, args.channel_mask, args.alpha, args.seed)


