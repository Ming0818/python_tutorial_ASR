#!usr/bin/env python
# coding=utf-8
from Tkinter import mainloop

from Tkinter import StringVar

from Tkinter import Entry

from Tkinter import Label

from Tkinter import Tk
from util import my_button, center_window
import datetime
now_date=datetime.datetime.now().strftime("%Y-%m-%d")
class Count_days:
    def getYmD(self,string_list):
        Y = int(string_list[0])
        m = int(string_list[1])
        D = int(string_list[2])
        return Y, m, D


    def get3int(self,date_start):
        date_start=str(date_start)
        if '-' in date_start:
            startY, startm, startD = self.getYmD(date_start.split('-'))
        elif '/' in date_start:
            startY, startm, startD = self.getYmD(date_start.split('/'))
        elif '.' in date_start:
            startY, startm, startD = self.getYmD(date_start.split('.'))
        else:
            print date_start
            startY = int(date_start[0:4])
            startm = int(date_start[4:6])
            startD = int(date_start[6:8])
        return startY, startm, startD


    def count_days(self,date_start="2016-01-03", date_end="2016-03-03"):
        date_start=self.contends_start.get()
        date_end=self.contends_end.get()
        print date_start
        print date_end
        endY, endm, endD = self.get3int(date_end)
        sY, sm, sD = self.get3int(date_start)
        end = datetime.date(endY, endm, endD)
        s = datetime.date(sY, sm, sD)
        total = (end - s).days
        self.contends_tot.set(total)
        return total


    def __init__(self):
        root = Tk()
        root.title('count days')
        root.resizable(False, False)
        center_window(root, 200, 200)
        label_start=Label(root,text="start date:")
        label_start.pack()
        entry_start=Entry(root)
        entry_start.pack()
        self.contends_start=StringVar()
        entry_start.config(textvariable=self.contends_start)
        self.contends_start.set('2014-07-07')
        label_end = Label(root, text="end date:")
        label_end.pack()
        entry_end = Entry(root)

        entry_end.pack()
        self.contends_end = StringVar()
        self.contends_end.set(now_date)
        entry_end.config(textvariable=self.contends_end)
        label_tot = Label(root, text="total days:")
        label_tot.pack()
        entry_tot = Entry(root)
        entry_tot.pack()
        self.contends_tot = StringVar()
        entry_tot.config(textvariable=self.contends_tot, state='disabled', highlightbackground='red')
        my_button(root, "count days:", "clik to count", self.count_days)

Count_days()
mainloop()
