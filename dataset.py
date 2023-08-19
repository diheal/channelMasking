import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def get_data(dir,data_name,transformer=None,divide_seed=None):
    if not data_name in ['uschad','ucihar','mobiact','motion']:
        raise ValueError("this dataset is not exist")
    if data_name == 'uschad':
        x_train = np.load('{}/USCHAD_X_Train_1s.npy'.format(dir))
        y_train = np.load('{}/USCHAD_Y_Train_1s.npy'.format(dir))
        x_test = np.load('{}/USCHAD_X_Test_1s.npy'.format(dir))
        y_test = np.load('{}/USCHAD_Y_Test_1s.npy'.format(dir))
        x_val = np.load('{}/USCHAD_X_Val_1s.npy'.format(dir))
        y_val = np.load('{}/USCHAD_Y_Val_1s.npy'.format(dir))

        if transformer:
            std = StandardScaler()
            numbers, n_timesteps, n_features, n_outputs = x_train.shape[0], x_train.shape[1], x_train.shape[2], \
                                                          y_train.shape[1]
            x_train = np.reshape(x_train, (-1, n_features))
            x_train = std.fit_transform(x_train)
            x_train = np.reshape(x_train, (numbers, n_timesteps, n_features))

            test_num = x_test.shape[0]
            x_test = np.reshape(x_test, (-1, n_features))
            x_test = std.transform(x_test)
            x_test = np.reshape(x_test, (test_num, n_timesteps, n_features))

            val_num = x_val.shape[0]
            x_val = np.reshape(x_val, (-1, n_features))
            x_val = std.transform(x_val)
            x_val = np.reshape(x_val, (val_num, n_timesteps, n_features))

        return x_train, y_train, x_val, y_val, x_test, y_test


    elif data_name=='ucihar':
        x_data = np.load('{}/UCI_X.npy'.format(dir))
        y_data = np.load('{}/UCI_Y.npy'.format(dir))
        subject_index = np.load('{}/UCI_Subject.npy'.format(dir))
        subject_list = np.arange(1, 31)
        np.random.seed(888)
        np.random.shuffle(x_data)
        np.random.seed(888)
        np.random.shuffle(y_data)
        np.random.seed(888)
        np.random.shuffle(subject_index)

        divide = divide_seed
        if divide == 1:
            test_subject = set([1, 2, 3, 4, 5, 6])
            train_subject = np.arange(13, 31).tolist()
            val_subject = [7, 8, 9, 10, 11, 12]
        elif divide == 2:
            test_subject = set([25, 26, 27, 28, 29, 30])
            train_subject = np.arange(1, 19)
            val_subject = np.arange(19, 25)
        elif divide == 3:
            test_subject = set(np.arange(7, 13))
            train_subject = np.concatenate((np.arange(1, 7), np.arange(13, 25)), axis=0)
            val_subject = np.arange(25, 31)
        elif divide == 4:
            test_subject = set(np.arange(13, 19))
            train_subject = np.concatenate((np.arange(1, 13), np.arange(25, 31)), axis=0)
            val_subject = np.arange(19, 25)
        elif divide == 5:
            test_subject = set(np.arange(19, 25))
            train_subject = np.concatenate((np.arange(7, 19), np.arange(25, 31)), axis=0)
            val_subject = np.arange(1, 7)
        else:
            train_val_subject, test_subject = train_test_split(subject_list, test_size=0.2, random_state=divide)
            train_subject, val_subject = train_test_split(train_val_subject, test_size=0.2, random_state=divide)
            test_subject = set(test_subject)
        train_subject = set(list(train_subject))

        train_bool = [i in train_subject for i in subject_index.flatten()]
        x_train = x_data[train_bool]
        y_train = y_data[train_bool]
        x_train_sub = subject_index[train_bool]

        test_bool = [i in test_subject for i in subject_index.flatten()]
        x_test = x_data[test_bool]
        y_test = y_data[test_bool]

        val_bool = [i in val_subject for i in subject_index.flatten()]
        x_val = x_data[val_bool]
        y_val = y_data[val_bool]

        return x_train,y_train,x_val, y_val,x_test,y_test

    elif data_name == 'motion':
        x_data = np.load('{}/Mo_X_1s.npy'.format(dir))
        y_data = np.load('{}/Mo_Y_1s.npy'.format(dir))
        subject_index = np.load('{}/Mo_Sub_1s.npy'.format(dir))  # 1-24
        np.random.seed(888)
        np.random.shuffle(x_data)
        np.random.seed(888)
        np.random.shuffle(y_data)
        np.random.seed(888)
        np.random.shuffle(subject_index)
        subject_list = np.unique(subject_index)
        n_timesteps, n_features, n_outputs = x_data.shape[1], x_data.shape[2], y_data.shape[1]

        divide = divide_seed
        if divide == 1:
            test_subject = set([1, 2, 3, 4, 5])
            train_subject = np.arange(10, 25).tolist()
            val_subject = [7, 8, 9, 10, 11, 12]
        elif divide == 2:
            test_subject = set([20, 21, 22, 23, 24])
            train_subject = np.arange(1, 16)
            val_subject = np.arange(20, 25)
        elif divide == 3:
            test_subject = set([6, 7, 8, 9, 10])
            train_subject = np.concatenate((np.arange(1, 6), np.arange(11, 21)), axis=0)
            val_subject = np.arange(21, 25)
        elif divide == 4:
            test_subject = set([11, 12, 13, 14, 15])
            train_subject = np.concatenate((np.arange(1, 11), np.arange(16, 21)), axis=0)
            val_subject = np.arange(21, 25)
        elif divide == 5:
            test_subject = set([16, 17, 18, 19])
            train_subject = np.arange(1, 16)
            val_subject = np.arange(20, 25)
        else:
            train_val_subject, test_subject = train_test_split(subject_list, test_size=0.2, random_state=divide)
            train_subject, val_subject = train_test_split(train_val_subject, test_size=0.2, random_state=divide)
            test_subject = set(test_subject)
        train_subject = set(list(train_subject))

        train_bool = [i in train_subject for i in subject_index.flatten()]
        x_train = x_data[train_bool]
        y_train = y_data[train_bool]

        test_bool = [i in test_subject for i in subject_index.flatten()]
        x_test = x_data[test_bool]
        y_test = y_data[test_bool]

        val_bool = [i in val_subject for i in subject_index.flatten()]
        x_val = x_data[val_bool]
        y_val = y_data[val_bool]

        if transformer:
            std = StandardScaler()
            x_train_num = x_train.shape[0]
            x_train = np.reshape(x_train, (-1, n_features))
            x_train = std.fit_transform(x_train)
            x_train = np.reshape(x_train, (x_train_num, n_timesteps, n_features))

            x_test_num = x_test.shape[0]
            x_test = np.reshape(x_test, (-1, n_features))
            x_test = std.transform(x_test)
            x_test = np.reshape(x_test, (x_test_num, n_timesteps, n_features))

            x_val_num = x_val.shape[0]
            x_val = np.reshape(x_val, (-1, n_features))
            x_val = std.transform(x_val)
            x_val = np.reshape(x_val, (x_val_num, n_timesteps, n_features))

        return x_train, y_train, x_val, y_val, x_test, y_test




