#!usr/bin/env python
# coding=gbk
import tkFileDialog
import wave
from Tkconstants import LEFT, SUNKEN, X, RIGHT
from vote_result import use_the_ASR
from datetime import datetime
from Tkinter import mainloop

from Tkinter import StringVar

from Tkinter import Entry

from Tkinter import Frame

from Tkinter import Tk

from Tkinter import Button

from Tkinter import Label
from pyaudio import PyAudio, paInt16

from classification_SGD import start_calssification_SGD

# define of params
from util import center_window

NUM_SAMPLES = 2000
framerate = 8000
channels = 1
sampwidth = 2
# record time
TIME = 10


class show_the_project:
    def start_identification(self):
        self.contents_result_entry.set(' ')
        path = self.contents_wave_path.get()
        speaker_name = use_the_ASR(path)  # start identification
        whose_vale = 'Hi, ' + speaker_name + ' !'
        self.contents_result_entry.set(whose_vale)
        return 0

    def save_wave_file(self, filename, data):
        '''save the date to the wav file'''
        wf = wave.open(filename, 'wb')
        wf.setnchannels(channels)
        wf.setsampwidth(sampwidth)
        wf.setframerate(framerate)
        wf.writeframes("".join(data))
        wf.close()
        return 0

    def my_button(self, root, label_text, button_text, button_func):
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
        return 0

    def record_wave(self):
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
            print 'recording.....',
            print (count * 100) / (TIME * 4), '/', 100

        filename = datetime.now().strftime("%Y-%m-%d_%H_%M_%S") + ".wav"
        self.save_wave_file(filename, save_buffer)
        save_buffer = []
        print filename, "saved"
        self.contents_wave_path.set(filename)
        return 0

    def choose_file(self):
        self.contents_wave_path.set("choosing now")
        filename = tkFileDialog.askopenfilename()
        print filename
        self.contents_wave_path.set(filename)
        return 0

    def __init__(self):
        root = Tk()
        root.title('speaker identification')
        root.resizable(False, False)
        center_window(root, 300, 250)
        separator = Frame(root, height=4, bd=2, relief=SUNKEN)
        separator.pack(fill=X, padx=5, pady=5)

        lable_title = Label(root, text="说话人识别".decode('gbk').encode('utf-8'))
        lable_title.pack()

        separator = Frame(root, height=2, bd=1, relief=SUNKEN)
        separator.pack(fill=X, padx=5, pady=5)

        frm = Frame(root)

        frm_left = Frame(frm)
        # label details
        label = Label(frm_left)
        label['text'] = "Record a wave"
        label.pack()
        # label details
        button = Button(frm_left)
        button['text'] = "clik to record"
        button['command'] = self.record_wave
        button.pack()
        #
        # record_wave_path=Entry(frm_left)
        # record_wave_path.pack()
        frm_left.pack(side=LEFT)

        frm_right = Frame(frm)
        self.my_button(frm_right, "choose a wav", "click and find", self.choose_file)
        # choose_wave_path=Entry(frm_right)
        # choose_wave_path.pack()
        frm_right.pack(side=LEFT)

        frm.pack()

        separator = Frame(root, height=2, bd=1, relief=SUNKEN)
        separator.pack(fill=X, padx=5, pady=5)

        frm_path = Frame(root)
        labe_path = Label(frm_path)
        labe_path['text'] = 'file path:'
        labe_path.pack(side=LEFT)

        wave_path = Entry(frm_path, state='disabled')
        wave_path.pack(side=RIGHT)
        # show your wav file path
        self.contents_wave_path = StringVar()
        # self.contents_wave_path.set("path show out")
        wave_path.config(textvariable=self.contents_wave_path)

        frm_path.pack()
        separator = Frame(root, height=2, bd=1, relief=SUNKEN)
        separator.pack(fill=X, padx=5, pady=5)

        # 开始检测
        frm_start = Frame(root)
        self.my_button(frm_start, "start", "clik to start", self.start_identification)
        frm_start.pack()
        separator = Frame(root, height=2, bd=1, relief=SUNKEN)
        separator.pack(fill=X, padx=5, pady=5)
        # 显示结果输出
        frm_result = Frame(root)
        result_label = Label(frm_result, text='识别结果:'.decode('gbk').encode('utf-8'))
        result_label.pack(side=LEFT)

        result_entry = Entry(frm_result, state='disabled', highlightbackground='red')
        result_entry.pack(side=RIGHT)
        # show result contends
        self.contents_result_entry = StringVar()
        self.contents_result_entry.set(" ".decode('gbk').encode('utf-8'))
        result_entry.config(textvariable=self.contents_result_entry)

        frm_result.pack()
        separator = Frame(root, height=2, bd=1, relief=SUNKEN)
        separator.pack(fill=X, padx=5, pady=5)

    def my_button(self, root, label_text, button_text, button_func):
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
        return 0


def main_of_me():
    show_the_project()

    mainloop()
    return 0


if __name__ == "__main__":
    main_of_me()
