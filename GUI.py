#!usr/bin/env python3

from tkinter import *
import tkinter as tk
from PIL import Image,ImageOps, ImageTk
import PIL.Image
from tkinter import ttk,filedialog
import numpy as np
import cv2
import matplotlib.pyplot as plt
from glcm import * # outputs#,GLCM_FeatureExtraction_0,GLCM_FeatureExtraction_45,GLCM_FeatureExtraction_90,GLCM_FeatureExtraction_135,GLCM_FeatureExtraction_180,GLCM_FeatureExtraction_270
from sklearn.cluster import KMeans

root = Tk()
root.title('Texture Segmentation using GLCM Features')
# root.iconbitmap('logo.png')
root.configure(background='#FFEE5C')

# final_image = np.zeros((50,50))
         
def open():
    global display
    global img_array
    global display_image

    root.filename = filedialog.askopenfilename(
                                                initialdir="",
                                                title="Select file",
                                                filetypes=(("jpeg", "*.jpeg"),("jpg", "*.jpg"),("png","*.png"),("all files","*.*"))
                                               )
    pil_image = PIL.Image.open(root.filename)
    display_image = pil_image
    size = (50,50)
    pil_image = pil_image.resize(size)
    pil_image = ImageOps.grayscale(pil_image) 
    img_array = np.array(pil_image)
    print(img_array.shape)
    print(type(img_array))
    window1(img_array)
    return img_array
    # plt.imshow(pil_image)
    # return img_gray
    
def list_to_PIL(image):
    new_width = 0
    new_height = 0
    image = np.array(image,dtype=np.uint8)
    if len(image.shape) == 3:
        image = np.flip(image,axis=-1)
    image = PIL.Image.fromarray(image)  
    image = np.array(image,dtype=np.uint8)
    if len(image.shape) == 3:
        image = np.flip(image,axis=-1)
    image = PIL.Image.fromarray(image)  
    width,height = image.size
    if width<height:
        new_height = 300
        new_width = (width/height)*300 

    elif width>height:
        new_width = 300
        new_height = ((width/height)**-1)*300

    elif height==width:
        new_width = 300
        new_height = 300 
    size = (int(new_width), int(new_height))

    image = image.resize(size)
    return ImageTk.PhotoImage(image)

def window1(image):
    global display
    global image_label
    global img_array
    global pil_image
    global display_image
    global entry1
    global entry2
    
    image_label.grid_forget()
    window1 = tk.Toplevel()
    window1.title('Texture Segmentation using GLCM Features')
    # window1.iconbitmap('logo.png')
    window1.configure(background='#077089')
    
    # image_label = tk.Label(window1)
    # image_label.grid(column=0, row=4, padx=100, pady=4)

    heading = Label(window1,text="Texture Segmentation using GLCM Features\n", font=("times", 15,"bold"),bg='#077089',fg="white",relief=FLAT)
    heading.grid(column=2, row=0, padx=4, pady=4)


    btn1 = tk.Button(window1, text="Show GLCM Features", width=30, command= lambda: outputs(image,filter_size,k),relief=FLAT)
    btn1.grid(column=3, row=1, padx=10, pady=10)
 
    btn3 = tk.Button(window1, text="Show Output", width=30, command=lambda: show_image() ,relief=FLAT)
    btn3.grid(column=1, row=1, padx=10, pady=10)

    
    display = list_to_PIL(display_image)
    image_label = tk.Label(window1, image=display)
    image_label.grid(column=2, row=4,padx=10, pady=10)
    
    canvas1 = tk.Canvas(window1, width = 300, height =150, bg ='#F0F0F0' )
    canvas1.grid(column=1,row=6,padx=100,pady=8)
    
    label1 = tk.Label(window1, text='Please enter the Filter size you want',bg='#F0F0F0',fg="black")
    label1.config(font=('times', 10))
    
    canvas1.create_window(150, 25, window=label1)
    entry1 = (tk.Entry (window1)) 
    canvas1.create_window(150, 65, window=entry1)
    
    canvas2 = tk.Canvas(window1, width = 300, height =150, bg ='#F0F0F0' )
    canvas2.grid(column=3,row=6,padx=100,pady=8)
    
    label2 = tk.Label(window1, text='Please enter the desired no. of clusters',bg='#F0F0F0',fg="black")
    label2.config(font=('times', 10))
    
    canvas2.create_window(150, 25, window=label2)
    entry2 = (tk.Entry (window1)) 
    canvas2.create_window(150, 65, window=entry2)
    
    btn_ok1 = tk.Button(window1,text="Confirm?",width=7,bg="#F0F0F0", fg="black",command= lambda:get_input(),  font=("times",10))
    canvas2.create_window(150,100, window=btn_ok1)
                                                 
    btn_save = tk.Button(window1, text='Save to disk', width=14, command=lambda: savefile(myimg))
    btn_save.grid(column=1, row=9, padx=10, pady=10)

    btn_exit = tk.Button(window1,text="Exit",width=14,bg="#FF665C", fg="black",command=lambda: window1.destroy(),  font=("times", 10))
    btn_exit.grid(column=3,row=9,padx=10,pady=10)

    credits = Label(window1,text="\nA project by\nSwapnil Joshi, Vishvesh Kodihal & Aniket Giri", font=("times", 10),bg='#077089',fg="white")
    credits.grid(column=2, row=15, padx=4, pady=10)
    
                                                             
def get_input():  
    global filter_size
    global k
    filter_size = int(entry1.get())
    k = int(entry2.get())
    print(filter_size,k)



def show_image():
    global display2
    global final_image    
    window2 = tk.Toplevel()
    window2.title('Texture Segments')
    # window1.iconbitmap('logo.png')
    window2.configure(background='#077089')

    a = just_for_returning()
    print(a)
    print(type(a))
    display2 = list_to_PIL(a)
    image_label = tk.Label(window2, image=display2)
    image_label.grid(column=2, row=4,padx=10, pady=10)
    # plt.imshow(a)
    #plot 1:
    plt.subplot(2, 2, 2)
    # plt.savefig('plot.png', dpi=300, bbox_inches='tight')
    plt.imshow(a)
    
    plt.title("Texture Segments")
    
heading = Label(root,text="Texture Segmentation using GLCM Features\n", font=("times", 15,"bold"),bg='#FFEE5C',fg="black")
heading.grid(column=0,row=1,columnspan=9999,padx=100,pady=2,sticky="")

credits = Label(root,text="\nA project by\nSwapnil Joshi, Vishvesh Kodihal & Aniket Giri", font=("times", 10),bg='#FFEE5C',fg="black")
credits.grid(column=0,row=13,columnspan=9999,padx=100,pady=1)

btn_open = tk.Button(root,text="Choose a File",command=lambda:open(),width=14,bg="#5C88FF", fg="white",font=("times", 10), relief=FLAT)
btn_open.grid(column=0,row=2,columnspan=9999,padx=100,pady=8)


image_label = tk.Label(root)
#image_label.grid(column=0, row=4, columnspan=9999, padx=100, pady=4)

btn_exit = tk.Button(root,text="Exit",width=14,bg="#FF665C", fg="black",command=lambda: root.destroy(),  font=("times", 10))
btn_exit.grid(column=0,row=11,columnspan=9999,padx=100,pady=14)




root.mainloop()
