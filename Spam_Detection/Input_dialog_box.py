import tkinter as tk
import pickle
from naive_bayes import process_message, SpamClassifier
root = tk.Tk()
fundo = tk.PhotoImage(file="down.png")
w = fundo.width()
h = fundo.height()
canvas1 = tk.Canvas(root, width = 450, height = 250,bg='black')
canvas1.pack(side='top', fill='both', expand='yes')
canvas1.create_image(0,0, image=fundo, anchor='nw')
entry1 = tk.Entry (root) 
canvas1.create_window(200, 140, window=entry1)

#This is the link of the hdf5 file that was created due to training, now you can fetch results directly from the model
with open("C:/Users/RAHUL GUPTA/Desktop/Spam_Detection/model.hdf5", "rb") as f:
    mp = pickle.load(f)

def value():
    new = entry1.get()
    new = process_message(new)
    str1="This is a Spam Mail"
    str2="This is NOT a Spam Mail"
    if(mp.classify(new) == True):
        widget = tk.Label(canvas1, text=str1, fg='white', bg='black')
        widget.place(x=20, y=60)
        widget.pack()
        canvas1.create_window(220, 242, window=widget)  
    else:
        widget = tk.Label(canvas1, text=str2, fg='white', bg='black')
        widget.place(x=20, y=60)
        widget.pack()
        canvas1.create_window(220, 242, window=widget)  

button1 = tk.Button(text='Enter', command = value)
canvas1.create_window(200, 180, window=button1)

root.mainloop()
