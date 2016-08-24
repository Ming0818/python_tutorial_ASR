# !/usr/bin/python2
# coding=utf-8


from Tkinter import Label, Button

import pylab
import scipy
import scipy.io.wavfile as wf
import spectrum
from numpy import array
from numpy.random import shuffle
import wave
from datetime import datetime
from pyaudio import PyAudio, paInt16


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


def shuff_input_data(inputXs=array([1, 2, 3, 4]), outputYs=array([2, 3, 4, 5]), shuffled=True):
    row_num = len(inputXs)
    idx = array(range(row_num))
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
