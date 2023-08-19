import numpy as np
import tensorflow as tf

def get_angles(pos, i, d_model):
  angle_rates = 1 / np.power(10000, (2 * (i//2)) / np.float32(d_model))
  return pos * angle_rates

def positional_encoding(position, d_model):
  angle_rads = get_angles(np.arange(position)[:, np.newaxis],
                          np.arange(d_model)[np.newaxis, :],
                          d_model)

  angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])

  angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])

  pos_encoding = angle_rads[np.newaxis, ...]

  return tf.cast(pos_encoding, dtype=tf.float32)

def span_mask(seq_len, max_gram=3, p=0.2, goal_num_predict=15):
    ngrams = np.arange(1, max_gram + 1, dtype=np.int64)
    pvals = p * np.power(1 - p, np.arange(max_gram))
    # alpha = 6
    # pvals = np.power(alpha, ngrams) * np.exp(-alpha) / factorial(ngrams)# possion
    pvals /= pvals.sum(keepdims=True)
    mask_pos = set()
    while len(mask_pos) < goal_num_predict:
        n = np.random.choice(ngrams, p=pvals)
        n = min(n, goal_num_predict - len(mask_pos))
        anchor = np.random.randint(seq_len)
        if anchor in mask_pos:
            continue
        for i in range(anchor, min(anchor + n, seq_len - 1)):
            mask_pos.add(i)
    return list(mask_pos)

def semi_record(data_name,type,time_mask,channel_mask,alpha,number,value,divide=None,tips=None,beta = None,epoch=None):
    if data_name=='uschad':
        if type=='time':
            suf = 'semi_time{}_number{} f1: {}'.format(time_mask,number,value)
        elif type == 'spantime':
            suf = 'semi_spantime{}_number{} f1: {}'.format(time_mask,number,value)
        elif type == 'spantime_channel':
            suf = 'semi_spantime{}_channel{}_alpha{}_number{} f1: {}'.format(time_mask,channel_mask,alpha,number,value)
        elif type == 'time_channel':
            suf = 'semi_time{}_channel{}_alpha{}_number{} f1: {}'.format(time_mask, channel_mask, alpha,number, value)
        elif type == 'channel':
            suf = 'semi_channel{}_number{} f1: {}'.format(channel_mask,number, value)
        else:
            raise ValueError("the type is not exist")
    else:
        if type=='time':
            suf = 'semi_time{}_divide{}_number{} f1: {}'.format(time_mask,divide,number,value)
        elif type == 'spantime':
            suf = 'semi_spantime{}_divide{}_number{} f1: {}'.format(time_mask,divide,number,value)
        elif type == 'spantime_channel':
            suf = 'semi_spantime{}_channel{}_alpha{}_divide{}_number{} f1: {}'.format(time_mask,channel_mask,alpha,divide,number,value)
        elif type == 'time_channel':
            suf = 'semi_time{}_channel{}_alpha{}_divide{}_number{} f1: {}'.format(time_mask, channel_mask, alpha,divide,number, value)
        elif type == 'channel':
            suf = 'semi_channel{}_divide{}_number{} f1: {}'.format(channel_mask,divide,number, value)
        else:
            raise ValueError("the type is not exist")

    if tips != None:
        suf += tips
    if beta != None:
        suf += '_beta{}'.format(beta)
    if epoch != None:
        suf += '_epoch{}'.format(epoch)
    object = None
    if data_name=='uschad':
        object = open("uschad_record.txt", "a+")
    elif data_name == 'ucihar':
        object = open("ucihar_record.txt", "a+")
    elif data_name == 'mobiact':
        object = open("act_record.txt", "a+")
    elif data_name == 'motion':
        object = open("motion_record.txt", "a+")
    object.write("\n")
    object.write(suf)
    object.write("\n")
    object.close()

def evaluate_record(data_name,type,time_mask,channel_mask,alpha,beta,value,divide=None,epoch=None,discriminate=None,tips=None):
    if data_name=='uschad':
        if type=='time':
            suf = 'time{} f1: {}'.format(time_mask,value)
        elif type == 'spantime':
            suf = 'spantime{} f1: {}'.format(time_mask,value)
        elif type == 'spantime_channel':
            suf = 'spantime{}_channel{}_alpha{} f1: {}'.format(time_mask,channel_mask,alpha,value)
        elif type == 'time_channel':
            suf = 'time{}_channel{}_alpha{} f1: {}'.format(time_mask, channel_mask, alpha, value)
        elif type == 'channel':
            suf = 'channel{} f1: {}'.format(channel_mask, value)
        else:
            raise ValueError("the type is not exist")
    else:
        if type=='time':
            suf = 'time{}_divide{} f1: {}'.format(time_mask,divide,value)
        elif type == 'spantime':
            suf = 'spantime{}_divide{} f1: {}'.format(time_mask,divide,value)
        elif type == 'spantime_channel':
            suf = 'spantime{}_channel{}_alpha{}_divide{} f1: {}'.format(time_mask,channel_mask,alpha,divide,value)
        elif type == 'time_channel':
            suf = 'time{}_channel{}_alpha{}_divide{} f1: {}'.format(time_mask, channel_mask, alpha,divide, value)
        elif type == 'channel':
            suf = 'channel{}_divide{} f1: {}'.format(channel_mask,divide, value)
        else:
            raise ValueError("the type is not exist")
    suf = "evaluate_" + suf
    if epoch!=150:
        suf += '_epoch{}'.format(epoch)
    if beta != None:
        suf += '_beta{}'.format(beta)
    if discriminate!=None and discriminate:
        suf += '_dis_beta{}'.format(beta)
    if tips!=None:
        suf += tips
    object = None
    if data_name=='uschad':
        object = open("uschad_record.txt", "a+")
    elif data_name == 'ucihar':
        object = open("ucihar_record.txt", "a+")
    elif data_name == 'mobiact':
        object = open("act_record.txt", "a+")
    elif data_name == 'motion':
        object = open("motion_record.txt", "a+")
    object.write("\n")
    object.write(suf)
    object.write("\n")
    object.close()

def save_model(data_name,my_type,time_mask,channel_mask,alpha,divide,model,epoch=None):
    # if i > int(epoch * 2 // 3) and epoch_loss_last < cur_loss:
    #     pass
    model_dir = ''
    if data_name == 'uschad':
        if my_type == 'time':
            model_dir = 'model/{}/time{}'.format(data_name, time_mask)
        elif my_type == 'spantime':
            model_dir = 'model/{}/spantime{}'.format(data_name, time_mask)
        elif my_type == 'spantime_channel':
            model_dir = 'model/{}/spantime{}_channel{}_alpha{}'.format(data_name, time_mask, channel_mask, alpha)
        elif my_type == 'time_channel':
            model_dir = 'model/{}/time{}_channel{}_alpha{}'.format(data_name, time_mask, channel_mask,
                                                                   alpha)
        elif my_type == 'channel':
            model_dir = 'model/{}/channel{}'.format(data_name, channel_mask)
    else:
        if my_type == 'time':
            model_dir = 'model/{}/time{}_divide{}'.format(data_name, time_mask, divide)
        elif my_type == 'spantime':
            model_dir = 'model/{}/spantime{}_divide{}'.format(data_name, time_mask, divide)
        elif my_type == 'spantime_channel':
            model_dir = 'model/{}/spantime{}_channel{}_divide{}_alpha{}'.format(data_name, time_mask,
                                                                                channel_mask, divide, alpha)
        elif my_type == 'time_channel':
            model_dir = 'model/{}/time{}_channel{}_divide{}_alpha{}'.format(data_name, time_mask,
                                                                            channel_mask, divide, alpha)
        elif my_type == 'channel':
            model_dir = 'model/{}/channel{}_divide{}'.format(data_name, channel_mask, divide)

    if epoch != None and epoch != 150:
        model_dir += '_epoch{}'.format(epoch)

    tf.keras.models.save_model(model, model_dir)
    return model_dir

