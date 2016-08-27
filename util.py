# !/usr/bin/python2
# coding=utf-8
import os
import wave
from datetime import datetime
import pylab
import scipy
import scipy.io.wavfile as wf
import spectrum
from Tkinter import Label, Button
from numpy.random import shuffle
from pyaudio import PyAudio, paInt16
from mfcc_feature import mfcc


def load_data_X_Y(dir_path_to_dataset):
    print_true = True
    X = []  # array
    Y = []
    for parent, dirnames, filenames in os.walk(dir_path_to_dataset):

        for filename in filenames:
            if '.wav' in filename:
                wav_file_path = os.path.join(parent, filename)
                if 'adam_traba' in wav_file_path:
                    y = 1
                elif 'bartek_bulat' in wav_file_path:
                    y = 2
                elif 'damian_bulat' in wav_file_path:
                    y = 3
                elif 'katarzyna_konieczna' in wav_file_path:
                    y = 4
                elif 'konrad_malawski' in wav_file_path:
                    y = 5
                elif 'szczepan_bulat' in wav_file_path:
                    y = 6

                srate, wav_file_data = wf.read(wav_file_path)
                feat_mfcc_of_music = mfcc(wav_file_data, srate)
                # numpy.arry
                frames_total = len(feat_mfcc_of_music[:, 1])
                Y = Y + [y] * frames_total
                feat_mfcc_of_music_list = feat_mfcc_of_music.tolist()

                for frame in range(frames_total):
                    X = X + [feat_mfcc_of_music_list[frame]]

    return X, Y


def load_data_user_chose(wav_path_to_data):
    X = []

    srate, wav_file_data = wf.read(wav_path_to_data)
    feat_mfcc_of_music = mfcc(wav_file_data, srate)
    frames_total = len(feat_mfcc_of_music[:, 1])

    feat_mfcc_of_music_list = feat_mfcc_of_music.tolist()

    for frame in range(frames_total):
        X = X + [feat_mfcc_of_music_list[frame]]

    return X


def vote_the_max_times(data_pridect_Y):
    data_pridect_Y = data_pridect_Y.tolist()
    items = dict([(data_pridect_Y.count(i), i) for i in data_pridect_Y])
    max_times = (int(items[max(items.keys())]))
    return max_times


def get_speaker_name(vote):
    if vote == 1:
        pridect_name = 'adam'
    elif vote == 2:
        pridect_name = 'bartek'
    elif vote == 3:
        pridect_name = 'damian'

    elif vote == 4:
        pridect_name = 'katarzyna'
    elif vote == 5:
        pridect_name = 'konrad'
    elif vote == 6:
        pridect_name = 'szczepan'
    else:
        pridect_name = 'not exist in our system'
    return pridect_name


def my_button(root, label_text, button_text, button_func):
    '''''function of creat label and button'''
    # label details
    label = Label(root)
    label['text'] = label_text
    label.pack()
    # label details
    button = Button(root)
    button['text'] = button_text
    button['command'] = button_func
    button.pack()


def center_window(root, width, height):
    screenwidth = root.winfo_screenwidth()
    screenheight = root.winfo_screenheight()
    size = '%dx%d+%d+%d' % (width, height, (screenwidth - width) / 2, (screenheight - height) / 2)
    print(size)
    root.geometry(size)
    return 0


def save_wave_file(filename, data):
    # define of params
    framerate = 8000
    channels = 1
    sampwidth = 2
    '''save the date to the wav file'''
    wf = wave.open(filename, 'wb')
    wf.setnchannels(channels)
    wf.setsampwidth(sampwidth)
    wf.setframerate(framerate)
    wf.writeframes("".join(data))
    wf.close()
    return 0


def record_wave():
    # define of params
    NUM_SAMPLES = 2000
    framerate = 8000
    # record time
    TIME = 10
    # open the input of wave
    pa = PyAudio()
    stream = pa.open(format=paInt16, channels=1,
                     rate=framerate, input=True,
                     frames_per_buffer=NUM_SAMPLES)
    save_buffer = []
    count = 0
    while count < TIME * 4:
        # read NUM_SAMPLES sampling data
        string_audio_data = stream.read(NUM_SAMPLES)
        save_buffer.append(string_audio_data)
        count += 1
        print '.'

    filename = datetime.now().strftime("%Y-%m-%d_%H_%M_%S") + ".wav"
    save_wave_file(filename, save_buffer)
    save_buffer = []
    print filename, "saved"
    return 0


def shuffle_two_list_X_Y(X=[[1, 2], [2, 4]], Y=[3, 6]):
    len_list = len(Y)
    # print len_list
    index = [i for i in range(len_list)]
    shuffled_X = [1] * len_list
    shuffled_Y = [1] * len_list
    # print index
    shuffle(index)

    # print index
    for i in range(len(index)):
        shuffled_X[i] = X[index[i]]
        shuffled_Y[i] = Y[index[i]]

    return shuffled_X, shuffled_Y


def shuff_input_data(inputXs=scipy.array([1, 2, 3, 4]), outputYs=scipy.array([2, 3, 4, 5]), shuffled=True):
    row_num = len(inputXs)
    idx = scipy.array(range(row_num))
    shuffled and shuffle(idx)
    return (inputXs[idx], outputYs[idx])


def frame(x, fs, framesz, hop):
    framesamp = int(framesz * fs)
    hopsamp = int(hop * fs)
    w = scipy.hamming(framesamp)
    return scipy.array([w * x[i:i + framesamp]
                        for i in range(0, len(x) - framesamp, hopsamp)])


def stft(x, fs, framesz, hop):
    framesamp = int(framesz * fs)
    hopsamp = int(hop * fs)
    w = scipy.hamming(framesamp)
    X = scipy.array([scipy.fft(w * x[i:i + framesamp])
                     for i in range(0, len(x) - framesamp, hopsamp)])
    return X


def istft(X, fs, T, hop):
    x = scipy.zeros(T * fs)
    framesamp = X.shape[1]
    hopsamp = int(hop * fs)
    for n, i in enumerate(range(0, len(x) - framesamp, hopsamp)):
        x[i:i + framesamp] += scipy.real(scipy.ifft(X[n]))
    return x


def lpc(wav='example.wav'):
    (fs, sd) = wf.read(wav)
    sd = sd - scipy.mean(sd)
    sd /= scipy.amax(sd)
    lpcc, e = spectrum.lpc(sd, 12)
    pylab.plot(lpcc, label="lPCC")
    pylab.title('test')
    pylab.ylim(-5, 5)
    pylab.show()

    return 0
