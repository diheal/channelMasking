import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import *
import os
from utils import span_mask, save_model
from encoder import Encoder

def get_base(dir, data_name, my_type, time_mask, channel_mask, alpha, divide=None,epoch=None):
    if not os.path.exists(dir):
        raise ValueError("the path is not exist")
    dir_pre = os.path.join(dir, data_name)
    dir_suf = None
    if data_name == 'uschad':
        if my_type == 'time':
            dir_suf = 'time{}'.format(time_mask)
        elif my_type == 'spantime':
            dir_suf = 'spantime{}'.format(time_mask)
        elif my_type == 'spantime_channel':
            dir_suf = 'spantime{}_channel{}_alpha{}'.format(time_mask, channel_mask, alpha)
        elif my_type == 'time_channel':
            dir_suf = 'time{}_channel{}_alpha{}'.format(time_mask, channel_mask,
                                                                   alpha)
        elif my_type == 'channel':
            dir_suf = 'channel{}'.format(channel_mask)
        else:
            raise ValueError("the type is not exist")
    else:
        if my_type == 'time':
            dir_suf = 'time{}_divide{}'.format(time_mask, divide)
        elif my_type == 'spantime':
            dir_suf = 'spantime{}_divide{}'.format(time_mask, divide)
        elif my_type == 'spantime_channel':
            dir_suf = 'spantime{}_channel{}_divide{}_alpha{}'.format(time_mask, channel_mask, divide, alpha)
        elif my_type == 'time_channel':
            dir_suf = 'time{}_channel{}_divide{}_alpha{}'.format(time_mask,
                                                                            channel_mask, divide, alpha)
        elif my_type == 'channel':
            dir_suf = 'channel{}_divide{}'.format(channel_mask,divide)
        else:
            raise ValueError("the type is not exist")
    if epoch != 150:
        dir_suf += '_epoch{}'.format(epoch)

    print("the model path is: {}".format(os.path.join(dir_pre, dir_suf)))
    return tf.keras.models.load_model(os.path.join(dir_pre, dir_suf))

def get_evaluate(base, n_outputs):

    base = base.layers[0]
    base.trainable=False

    return tf.keras.models.Sequential(
        [base, GlobalAveragePooling1D(), Dense(units=256), BatchNormalization(), ReLU(), Dropout(0.1), \
         Dense(128), BatchNormalization(), ReLU(), Dropout(0.1), Dense(n_outputs, activation='softmax')])


def get_pretrain_model(num_layers, d_model, num_heads, dff, maximum_position_encoding, n_features):
    return tf.keras.models.Sequential(
        [Encoder(num_layers, d_model, num_heads, dff, maximum_position_encoding), Dense(256), \
         BatchNormalization(), ReLU(), Dropout(0.1), Dense(128), BatchNormalization(), ReLU(), Dropout(0.1),
         Dense(n_features)])


def pre_train(model, data_name, x_train, epoch, batch_size, optimizer, loss_func, my_type, n_timesteps, time_mask,
              n_features=None, channel_mask=None, alpha=None, divide=None):
    print("To begin the model, data:{} epoch:{} batchsize:{} type:{}".format(data_name, epoch, batch_size, my_type))
    seed = x_train.shape[0]
    cur_loss = 1e4
    for i in range(epoch):
        train_loss_dataset = tf.data.Dataset.from_tensor_slices(x_train).shuffle(seed,reshuffle_each_iteration=True).batch(batch_size)
        loss_batch = []
        for x in train_loss_dataset:
            x_np = x.numpy()

            time_index = None
            y_time = None
            if my_type in ['time','time_channel']:
                time_index = np.random.choice(n_timesteps, int(n_timesteps * time_mask * 0.01), replace=False)
                y_time = tf.convert_to_tensor(x_np[:, time_index, :], dtype=tf.float64)
                x_np[:, time_index, :] = 0
            elif my_type in ['spantime','spantime_channel']:
                time_index = span_mask(n_timesteps, goal_num_predict=int(n_timesteps * time_mask * 0.01))
                y_time = tf.convert_to_tensor(x_np[:, time_index, :], dtype=tf.float64)
                x_np[:, time_index, :] = 0

            y_channel, channel_index = None, None
            if my_type in ['spantime_channel','time_channel','channel']:
                channel_index = np.random.choice(n_features, channel_mask, replace=False)
                y_channel = tf.convert_to_tensor(x_np[:, :, channel_index], dtype=tf.float64)
                x_np[:, :, channel_index] = 0

            x_mask = tf.convert_to_tensor(x_np, dtype=tf.float64)

            loss = train_step(model, my_type, optimizer, loss_func, x_mask, y_time, time_index, y_channel, channel_index, alpha)

            loss_batch.append(loss.numpy())
        epoch_loss_last = np.mean(loss_batch)
        print('epoch:{} ==> loss:{}'.format(i + 1, epoch_loss_last))

        if i > int(epoch * 2 // 3) and epoch_loss_last < cur_loss:
            model_dir = save_model(data_name, my_type, time_mask, channel_mask, alpha, divide, model, epoch)
            cur_loss = epoch_loss_last
            print("epoch{} the model is saved in {}".format(i + 1, model_dir))

def train_step(model, my_type, optimizer, loss_func, x, y_time, time_index, y_channel=None, channel_index=None,
               alpha=None):
    with tf.GradientTape() as tape:
        out = model(x)
        if my_type in ['time','spantime','spantime_channel','time_channel']:
            y_t = tf.gather(out, time_index, axis=1)
        if my_type in ['channel','time_channel','spantime_channel']:
            y_c = tf.gather(out, channel_index, axis=2)

        if my_type in ['time','spantime']:
            loss = loss_func(y_time, y_t)
        elif my_type in ['channel']:
            loss = loss_func(y_channel, y_c)
        elif my_type in ['spantime_channel','time_channel']:
            alpha = alpha * 0.01
            loss = alpha * loss_func(y_time, y_t) + (1 - alpha) * loss_func(y_channel, y_c)

    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    return loss