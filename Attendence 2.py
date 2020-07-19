# -*- coding: utf-8 -*-
"""
Created on Sat Mar 14 01:18:02 2020

@author: KIIT
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Feb  6 13:58:28 2020

@author: KIIT
"""
from keras.models import Sequential
from keras.layers import Conv2D, ZeroPadding2D, Activation, Input, concatenate
from keras.models import Model
from keras.layers.normalization import BatchNormalization
from keras.layers.pooling import MaxPooling2D, AveragePooling2D
from keras.layers.merge import Concatenate
from keras.layers.core import Lambda, Flatten, Dense
from keras.initializers import glorot_uniform
from keras.engine.topology import Layer
from keras import backend as K
from keras.models import load_model
K.set_image_data_format('channels_first')


import tkinter as tk
from tkinter import Message,Text
import cv2,os
from PIL import Image,ImageTk
import pandas as pd
import numpy as np
import datetime
import time
from numpy import genfromtxt
import pickle
import tensorflow as tf
import tkinter.ttk as ttk
import tkinter.font as font
import csv
from utility import img_to_encoding,resize_img
from webcam_utility import detect_face,detect_face_realtime,find_face_realtime
from fr_utils import *
from inception_network import *

def triplet_loss(y_true, y_pred, alpha = 0.3):
        anchor = y_pred[0]
        positive = y_pred[1]
        negative = y_pred[2]
        pos_dist = tf.reduce_sum(tf.square(tf.subtract(anchor, positive)), axis=-1)
        neg_dist = tf.reduce_sum(tf.square(tf.subtract(anchor, negative)), axis=-1)
        basic_loss = tf.add(tf.subtract(pos_dist, neg_dist), alpha)
        loss = tf.reduce_sum(tf.maximum(basic_loss, 0.0))
        return loss

FRmodel = model(input_shape = (3,96,96))
FRmodel.compile(optimizer = 'adam', loss = triplet_loss, metrics = ['accuracy'])
load_weights_from_FaceNet(FRmodel) 

def ini_user_database():
        # check for existing database
        if os.path.exists(r'C:\Users\KIIT\Desktop\Deep Learning\Face Detection based Attendence\database\user_dict.pickle'):
            with open(r'C:\Users\KIIT\Desktop\Deep Learning\Face Detection based Attendence\database\user_dict.pickle', 'rb') as handle:
                user_db = pickle.load(handle)
        else:
            # make a new one
            # we use a dict for keeping track of mapping of each person with his/her face encoding
            user_db = {}

        return user_db

user_db = ini_user_database()

window=tk.Tk()
window.title("Face Recognizer")
window.geometry("1280x720")
dialog_title='QUIT'
dialog_text='Are you sure'
window.configure(background='grey')
window.grid_rowconfigure(0,weight=1)
window.grid_columnconfigure(0,weight=1)
message=tk.Label(window,text="FACE RECOGNITION BASED ATTENDENCE MANAGEMENT SYSTEM",bg='black',fg='white',width=100,height=5,font=('times',15))
message.place(x=230,y=20)
lbl=tk.Label(window,text='ENTER ID',width=30,height=3,bg='black',fg='white',font=('times',10))
lbl.place(x=100,y=200)
txt=tk.Entry(window,width=20,bg='black',fg='white',font=('times',10))
txt.place(x=350,y=220)
lbl2=tk.Label(window,text='ENTER NAME',width=30,height=3,bg='black',fg='white',font=('times',10))
lbl2.place(x=100,y=300)
txt2=tk.Entry(window,width=20,bg='black',fg='white',font=('times',10))
txt2.place(x=350,y=320)
lbl3=tk.Label(window,text='NOTIFICATION',width=30,height=3,bg='black',fg='white',font=('times',10))
lbl3.place(x=100,y=400)
message=tk.Label(window,text='',width=30,height=3,bg='black',fg='white',activebackground='black',font=('times',10))
message.place(x=350,y=400)
lbl4=tk.Label(window,text='ATTENDENCE',width=30,height=3,bg='black',fg='white',font=('times',10))
lbl4.place(x=100,y=600)
message2=tk.Label(window,text='',width=30,height=3,bg='black',fg='white',activebackground='black',font=('times',10))
message2.place(x=350,y=600)
def clear():
    txt.delete(0,'end')
    res=""
    message.configure(text=res)
def clear2():
    txt2.delete(0,'end')
    res=""
    message2.configure(text=res)
def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        pass
    try:
        import unicodedata
        unicodedata.numeric(s)
        return True
    except (TypeError,ValueError):
        pass
    return False
        
def TakeImages():
    Id=(txt.get())
    name=(txt2.get())

    def add_user_img_path(user_db, FRmodel, name, img_path):
        if name not in user_db:
            user_db[name] = img_to_encoding(img_path, FRmodel)
            # save the database
            with open(r'C:\Users\KIIT\Desktop\Deep Learning\Face Detection based Attendence\database\user_dict.pickle', 'wb') as handle:
                pickle.dump(user_db, handle, protocol=pickle.HIGHEST_PROTOCOL)
                res='User ' + name + ' added successfully'
                message.configure(text=res)
                
        else:
            res='The name is already registered! Try a different name.........'
            message.configure(text=res)
        
    def add_user_webcam(user_db, FRmodel, name,Id):
    # we can use the webcam to capture the user image then get it recognized
        face_found = detect_face(user_db, FRmodel,name,Id)
        print(face_found)
        if face_found:
            resize_img(r"C:\Users\KIIT\Desktop\Deep Learning\Face Detection based Attendence\Images\ "+name+"."+Id+".jpg")
            if name not in user_db:
                print("Face found......Adding User")
                add_user_img_path(user_db, FRmodel, name, r"C:\Users\KIIT\Desktop\Deep Learning\Face Detection based Attendence\Images\ "+name+"."+Id+".jpg")
            else:
                print('The name is already registered! Try a different name.........')
        else:
            print('There was no face found in the visible frame. Try again...........')
    
    add_user_webcam(user_db, FRmodel, name,Id)
    
    res="Images saved for ID:"+Id+" Name:"+name
    row=[Id,name]
    with open(r"C:\Users\KIIT\Desktop\Deep Learning\Face Detection based Attendence\Student Details\studentdetails.csv",'a+') as csvFile:
        writer=csv.writer(csvFile)
        writer.writerow(row)
    csvFile.close()
    message.configure(text=res)
    if(not is_number(Id) or not name.isalpha()):
        res='Enter fields correctly'
        message.configure(text=res)

def TrackImages():
    df=pd.read_csv(r"C:\Users\KIIT\Desktop\Deep Learning\Face Detection based Attendence\Student Details\studentdetails.csv")
    col_names=['Id','Name','Date','Time']
    attendence=pd.DataFrame(columns=col_names)
    check,identity=detect_face_realtime(user_db,FRmodel,threshold=0.6)
    if check:
        ts=time.time()
        date=datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d')
        timestamp=datetime.datetime.fromtimestamp(ts).strftime('%H:%M:%S')
        aa=df.loc[df['Name']==identity]['Id'].values
        attendence.loc[len(attendence)]=[aa,identity,date,timestamp]
        ts=time.time()
        date=datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d')
        timestamp=datetime.datetime.fromtimestamp(ts).strftime('%H:%M:%S')
        Hour,Minute,Second=timestamp.split(":")
        filename=r"C:\Users\KIIT\Desktop\Deep Learning\Face Detection based Attendence\Attendence\Attendence_"+date+"_"+Hour+"-"+Minute+"-"+Second+".csv"
        attendence.to_csv(filename,index=False)
        res=attendence
        message2.configure(text=res)
        
def clear_cache():
    os.remove(r'C:\Users\KIIT\Desktop\Deep Learning\Face Detection based Attendence\Image_to_be_recog\1.jpg')

clbutton=tk.Button(window,text='CLEAR',command=clear,width=20,height=3,bg='black',fg='white',activebackground='black',font=('times',10))
clbutton.place(x=500,y=200)
clbutton2=tk.Button(window,text='CLEAR',command=clear2,width=20,height=3,bg='black',fg='white',activebackground='black',font=('times',10))
clbutton2.place(x=500,y=300)
tkimage=tk.Button(window,text='TAKE IMAGES',command=TakeImages,width=20,height=3,bg='black',fg='white',activebackground='black',font=('times',10))
tkimage.place(x=140,y=500)
trainimage=tk.Button(window,text='CLEAR IMG',command=clear_cache,width=20,height=3,bg='black',fg='white',activebackground='black',font=('times',10))
trainimage.place(x=350,y=500)
trackimage=tk.Button(window,text='TRACK IMAGES',command=TrackImages,width=20,height=3,bg='black',fg='white',activebackground='black',font=('times',10))
trackimage.place(x=550,y=500)
quitwindow=tk.Button(window,text='QUIT',command=window.destroy,width=20,height=3,bg='black',fg='white',activebackground='black',font=('times',10))
quitwindow.place(x=750,y=500)
window.mainloop()