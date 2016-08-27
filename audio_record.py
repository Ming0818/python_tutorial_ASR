#!usr/bin/env python
# coding=utf-8
import wave
from Tkconstants import LEFT

from Tkinter import Tk

from Tkinter import Button

from Tkinter import Label
from datetime import datetime
from pyaudio import PyAudio, paInt16

# define of params
NUM_SAMPLES = 2000
framerate = 8000
channels = 1
sampwidth = 2
# record time
TIME = 10


def save_wave_file(filename, data):
    '''save the date to the wav file'''
    wf = wave.open(filename, 'wb')
    wf.setnchannels(channels)
    wf.setsampwidth(sampwidth)
    wf.setframerate(framerate)
    wf.writeframes("".join(data))
    wf.close()
    return 0


def my_button(root, label_text, button_text, button_func):
    '''''function of creat label and button'''
    # label details
    label = Label(root)
    label['text'] = label_text
    label.pack(side=LEFT)
    # label details
    button = Button(root)
    button['text'] = button_text
    button['command'] = button_func
    button.pack(side=LEFT)


def record_wave():
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


def main():
    root = Tk()
    my_button(root, "Record a wave", "clik to record", record_wave)
    root.mainloop()


if __name__ == "__main__":
    main()
