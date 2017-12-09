from os import chdir, listdir, path
import numpy as np
import csv
from random import shuffle
from math import ceil

class DatasetTool(object):
    def __init__(self, dataset_path, batch_size, timesteps):
        self.batch_size = batch_size
        self.timesteps = timesteps
        dataset_list = []
        #read csv path
        for emotion in listdir(dataset_path):
            for csvfile in listdir(emotion):
                dataset_list.append(path.abspath(csvfile))
        shuffle(dataset_list)
        csv_len = len(dataset_list)

        #split train and test set
        self.train_len = ceil(0.8 * csv_len)
        self.train_set = dataset_list[:self.train_len]
        self.test_len = csv_len - train_len
        self.test_set = dataset_list[self.train_len:]

        #calculate train and test batch number
        self.train_batch_len = self.train_len * 3 // batch_size
        self.test_batch_len = self.test_len * 3 // batch_size

        #init current batch pointer
        self.train_batch_pointer = 0
        self.test_batch_pointer = 0
        return

    def show_stat_info(self):
        print('train_len:', self.train_len * 3)
        print('test_len:', self.train_len * 3)
        print('train_batch_len:', self.train_batch_len)
        print('test_batch_len:', self.test_batch_len)
        return

    def emotion_dict(self, abs_path):
        if 'anger' in abs_path:
            return [1, 0, 0, 0, 0, 0]
        elif 'disgust' in abs_path:
            return [0, 1, 0, 0, 0, 0]
        elif 'fear' in abs_path:
            return [0, 0, 1, 0, 0, 0]
        elif 'happiness' in abs_path:
            return [0, 0, 0, 1, 0, 0]
        elif 'sadness' in abs_path:
            return [0, 0, 0, 0, 1, 0]
        elif 'surprise' in abs_path:
            return [0, 0, 0, 0, 0, 1]
        return -1

    def next_train_batch(self):
        '''
        return batch_x, batch_y
        batch_x = [[3s frame * Timestep] * batch_size]
        batch_y = [ground_truth * batch_size]
        ground_truth is the following number symbols:
            0 - anger
            1 - disgust
            2 - fear
            3 - happiness
            4 - sadness
            5 - surprise
        '''
        #fetch current batch interval and move the pointer
        pointer_interval = self.batch_size // 3
        batch_csv_list = self.train_set[self.train_batch_pointer : pointer_interval]
        self.train_batch_pointer += pointer_interval

        #fetch data
        logMel = []
        features = []
        ground_truth = []
        for csv_file_path in batch_csv_list:
            ground_truth.extend(emotion_dict(csv_file_path) for _ in range(3))
            with open(csv_file_path) as csv:
                reader = csv.reader(csv, delimiter=';')
                rows = [_ for _ in reader]
                logMel_rows = []
                feature_rows = []
                for row in rows:
                    logMel_rows.append(row[2 : 28])
                    temp = row[30 : 33]
                    temp.append(row[36])
                    feature_rows.append(temp)
                #3*297 lines, parse matrix
                for i in range(3):
                    one_logMel = np.array()
                    one_features = np.array()
                    mel_mat = np.array(logMel_rows[i * 297 + 1: (i + 1) * 297])
                    feature_mat = np.array(feature_rows[i * 297 + 1: (i + 1) * 297])
                    logMel.append(np.log(mel_mat))
                    features.append(feature_mat)

        return np.array(logMel), np.array(features), np.array(ground_truth)
        #return np.array([[np.zeros((400, 300, 3)) for __ in range(self.timesteps)] for _ in xrange(self.batch_size)]),\
          #np.array([[np.zeros((20)) for __ in range(self.timesteps)] for _ in xrange(self.batch_size)]),\
          #np.array([[1, 0, 0, 0, 0, 0] for _ in xrange(self.batch_size)])

    def next_test_batch(self):
        '''
        return test_batch_x, test_batch_y
        format is the same with next_train_batch
        '''
        return np.array([[np.zeros((400, 300, 3)) for __ in range(self.timesteps)] for _ in xrange(self.batch_size)]),\
          np.array([[np.zeros((20)) for __ in range(self.timesteps)] for _ in xrange(self.batch_size)]),\
          np.array([[1, 0, 0, 0, 0, 0] for _ in xrange(self.batch_size)])
