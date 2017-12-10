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
            for csvfile in listdir(dataset_path + '/' + emotion):
                dataset_list.append(dataset_path + '/' + emotion + '/' + csvfile)
        shuffle(dataset_list)
        csv_len = len(dataset_list)

        #split train and test set
        self.train_len = ceil(0.8 * csv_len)
        self.train_set = dataset_list[:self.train_len]
        self.test_len = csv_len - self.train_len
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

    def next_batch(self, is_train):
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
        if is_train:
            batch_csv_list = self.train_set[self.train_batch_pointer : pointer_interval]
            self.train_batch_pointer += pointer_interval
        else:
            batch_csv_list = self.test_set[self.test_batch_pointer : pointer_interval]
            self.test_batch_pointer += pointer_interval

        #fetch data
        logMel = []
        features = []
        ground_truth = []
        for csv_file_path in batch_csv_list:
            ground_truth.extend(self.emotion_dict(csv_file_path) for _ in range(3))
            with open(csv_file_path) as csvf:
                reader = csv.reader(csvf, delimiter=';')
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
                    mel_mat = np.array(logMel_rows[i * 297 + 1: (i + 1) * 297]).astype(np.float32)
                    feature_mat = np.array(feature_rows[i * 297 + 1: (i + 1) * 297]).astype(np.float32)
                    #print('before:\n', feature_mat)
                    #reduce mean, every 8 frames
                    feature_mat = np.reshape(feature_mat, (37, 8, 4))
                    #print('reshaped!:\n', feature_mat)
                    feature_mat = np.mean(feature_mat, axis = 1)
                    #print('mean!:\n', feature_mat)
                    logMel.append(np.log(mel_mat))
                    features.append(feature_mat)

        return np.array(logMel), np.array(features), np.array(ground_truth)

if __name__ == '__main__':
    dataset = DatasetTool('/scratch/user/liqingqing/info_concatenated', 3, 296)
    dataset.show_stat_info()
    batch_mel, batch_feature, batch_truth = dataset.next_batch(is_train = True)
    print('batch_mel_shape:', batch_mel.shape)
    print('batch_feature_shape', batch_feature.shape)
    print('batch_truth_shape', batch_truth.shape)
    print('\n')
    batch_mel, batch_feature, batch_truth = dataset.next_batch(is_train = False)
    print('test_batch_mel_shape:', batch_mel.shape)
    print('test_batch_feature_shape', batch_feature.shape)
    print('test_batch_truth_shape', batch_truth.shape)
