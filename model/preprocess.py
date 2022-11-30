import os
import csv
from intervaltree import IntervalTree
import h5py
import numpy as np

HOP_LENGTH = 512
WAV_SAMPLING_RATE_IN_HZ = 44100
META_PATH = '../data/musicnet/musicnet_metadata.csv'

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


def get_data(h5_path, label_path):
    trees = process_labels(label_path)
    one_hot_note = np.empty((128, 1))
    data_dict = np.empty((88, 1))
    length_dict = read_length()
    for item in os.listdir(h5_path):
        if not item.endswith('.h5'): continue
        uid = int(item[:-3])
        tree = trees[uid]
        seconds = length_dict[uid]
        f = h5py.File(os.path.join('',h5_path,item), 'r')
        frames = f['cqt'].shape[1]
        data_dict = np.concatenate([data_dict, np.array(f['cqt'])], axis=-1)
        notes = np.zeros((128, frames))
        for t in range(frames):
            l = int(t / frames * seconds * WAV_SAMPLING_RATE_IN_HZ)
            r = int((t + 1) / frames * seconds * WAV_SAMPLING_RATE_IN_HZ)
            for interval in tree[l:r]:
                note = interval[2][1]
                notes[note, t] = 1
        one_hot_note = np.concatenate([one_hot_note, notes], axis=-1)
    return data_dict, one_hot_note
    


def main():
    data, notes = get_data('../data/data_16K/test_data', '../musicnet/test_labels')
    print(data.shape, notes.shape)


if __name__ == '__main__':
    main()