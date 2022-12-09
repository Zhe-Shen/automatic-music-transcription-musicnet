import os
import csv
from intervaltree import IntervalTree
import h5py
import numpy as np

HOP_LENGTH = 512
WAV_SAMPLING_RATE_IN_HZ = 44100
META_PATH = '../data/musicnet/musicnet_metadata.csv'
WINDOW_SIZE = 64
NUM_FEATURES = 88
NOTE_SIZE = 128

def read_length():
    length_dict = dict()
    with open(META_PATH, 'r') as f:
        reader = csv.DictReader(f, delimiter=',')
        for label in reader:
            music_id = int(label['id'])
            seconds = int(label['seconds'])
            length_dict[music_id] = seconds
    return length_dict


def process_labels(path):
    trees = dict()
    for item in os.listdir(path):
        if not item.endswith('.csv'): continue
        uid = int(item[:-4])
        tree = IntervalTree()
        with open(os.path.join('',path,item), 'r') as f:
            reader = csv.DictReader(f, delimiter=',')
            for label in reader:
                start_time = int(label['start_time'])
                end_time = int(label['end_time'])
                instrument = int(label['instrument'])
                note = int(label['note'])
                start_beat = float(label['start_beat'])
                end_beat = float(label['end_beat'])
                note_value = label['note_value']
                tree[start_time:end_time] = (instrument,note,start_beat,end_beat,note_value)
        trees[uid] = tree
    return trees


def store_data(h5_path, label_path, is_training=False):
    trees = process_labels(label_path)
    one_hot_note = np.empty((NOTE_SIZE, 0))
    data_dict = np.empty((NUM_FEATURES, 0))
    length_dict = read_length()
    i = 0
    for item in os.listdir(h5_path):
        if not item.endswith('.h5'): continue
        uid = int(item[:-3])
        tree = trees[uid]
        seconds = length_dict[uid]
        f = h5py.File(os.path.join('',h5_path,item), 'r')
        frames = f['cqt'].shape[1]
        data = np.array(f['cqt'])
        if frames % WINDOW_SIZE > 0:
            # zero pad audio
            rmn = (frames // WINDOW_SIZE + 1) * WINDOW_SIZE - frames
            data = np.concatenate([data, np.zeros((NUM_FEATURES, rmn))], axis=-1)
            frames += rmn
        data_dict = np.concatenate([data_dict, data], axis=-1)
        notes = np.zeros((NOTE_SIZE, frames))
        for t in range(frames):
            l = int(t / frames * seconds * WAV_SAMPLING_RATE_IN_HZ)
            r = int((t + 1) / frames * seconds * WAV_SAMPLING_RATE_IN_HZ)
            for interval in tree[l:r]:
                note = interval[2][1]
                notes[note, t] = 1
        one_hot_note = np.concatenate([one_hot_note, notes], axis=-1)
        i += 1
        if i % 10 == 0:
            num = i // 10
            if is_training:
                np.save('./train/x_' + str(num) + '.npy', data_dict)
                np.save('./train/y_' + str(num) + '.npy', one_hot_note)
            else:
                np.save('./test/x_' + str(num) + '.npy', data_dict)
                np.save('./test/y_' + str(num) + '.npy', one_hot_note)
            one_hot_note = np.empty((NOTE_SIZE, 0))
            data_dict = np.empty((NUM_FEATURES, 0))
            
            
def load_data(num_list, is_training=False):
    one_hot_note = np.empty((NOTE_SIZE, 0))
    data_dict = np.empty((NUM_FEATURES, 0))
    if is_training:
        for i in num_list:
            path = './train/x_' + str(i) + '.npy'
            data = np.load(path)
            data_dict = np.concatenate([data_dict, data], axis=-1)
            path = './train/y_' + str(i) + '.npy'
            data = np.load(path)
            one_hot_note = np.concatenate([one_hot_note, data], axis=-1)
    else:
        for i in num_list:
            path = './test/x_' + str(i) + '.npy'
            data = np.load(path)
            data_dict = np.concatenate([data_dict, data], axis=-1)
            path = './test/y_' + str(i) + '.npy'
            data = np.load(path)
            one_hot_note = np.concatenate([one_hot_note, data], axis=-1)
    return data_dict, one_hot_note
    

def main():
    store_data('../data/data_16K/test_data', '../data/data_16K/test_labels')
    store_data('../data/data_16K/train_data', '../data/data_16K/train_labels', True)


if __name__ == '__main__':
    main()