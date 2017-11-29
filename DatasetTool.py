from os import chdir

class DatasetTool:
    def __init__(self, dataset_config_path, batch_size):
    '''
    dataset_config_path is the full path of dataset_config_file
    dataset_directory should looks like:
        -database_directory
            -subject1
                -anger
                    -sentence1
                        -noise1: this directory contains Timestep number of audio segments
                        ...
                        -noise20
                    ...
                    -sentence5
                -disgust
                ...
                -surprise
            -subject2
            ...
            -subjectx

    dataset_config_file should contain lines like below:
        sentence_path1 ground_truth1
        sentence_path2 ground_truth2
        ...
        sentence_path_n ground_truth_n

    CAUTION: the sequence of the list above should be random first! And it's sentence_path but audio_segment_path.
    At some arbitary ratio, the list above will be separated as train and test set in this __init__ function.

    '''


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

    def next_test_batch(self):
    '''
    return test_batch_x, test_batch_y
    format is the same with next_train_batch
    '''
