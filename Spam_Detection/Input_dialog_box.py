import tkinter as tk
import pickle
from naive_bayes import process_message, SpamClassifier

root = tk.Tk()

canvas1 = tk.Canvas(root, width = 400, height = 300)
canvas1.pack()

entry1 = tk.Entry (root) 
canvas1.create_window(200, 140, window=entry1)

#This is the link of the hdf5 file that was created due to training, now you can fetch results directly from the model
with open("C:/Users/RAHUL GUPTA/Desktop/Spam_Detection/model.hdf5", "rb") as f:
    mp = pickle.load(f)

def value():
    new = entry1.get()
    new = process_message(new)
    if(mp.classify(new) == True):
        print("This is a Spam Mail")
    else:
        print("This is NOT a Spam Mail")

button1 = tk.Button(text='Enter', command = value)
canvas1.create_window(200, 180, window=button1)

root.mainloop()