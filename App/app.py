import tkinter as tk
from PIL import ImageTk, Image
from tkinter import filedialog
import numpy as np
import tensorflow
from tensorflow.keras.preprocessing.image import img_to_array
from keras.models import model_from_json
from keras.preprocessing import image
import cv2

model = model_from_json(open("model/model.json", "r").read())
model.load_weights('model/model.h5')
face_haar_cascade = cv2.CascadeClassifier('model/haarcascade_frontalface_default.xml')
cap = cv2.VideoCapture(0)

live_tab = False
after_id = 0

def load_splash():
    global img
    for img_display in frame.winfo_children():
        img_display.destroy()
    img = Image.open("assets/splash.jpg")
    basewidth = 600
    wpercent = (basewidth / float(img.size[0]))
    hsize = int((float(img.size[1] * float(wpercent))))
    img = img.resize((basewidth, hsize), Image.ANTIALIAS)
    img = ImageTk.PhotoImage(img)
    panel_image = tk.Label(frame, image=img).pack()

def load_img():
    global img, img_data
    for img_display in frame.winfo_children():
        img_display.destroy()
    img_data = filedialog.askopenfilename(initialdir="/", title="Choose Image", filetypes=(("all files", "*.*"), ("png files", "*.png")))
    img = cv2.imread(img_data)
    img = emotion(img)
    img = image_resize(img)
    b,g,r = cv2.split(img)
    img = cv2.merge((r,g,b))
    im = Image.fromarray(img)
    img = ImageTk.PhotoImage(image=im)
    file_name = img_data.split('/')
    panel = tk.Label(frame, text=str(file_name[len(file_name) -1]).upper()).pack()
    panel_image = tk.Label(frame, image=img).pack()

def live_img():
    global img, img_data,after_id
    for img_display in frame.winfo_children():
        img_display.destroy()
    _, cam_img = cap.read()
    img = emotion(cam_img)
    b,g,r = cv2.split(img)
    img = cv2.merge((r,g,b))
    im = Image.fromarray(img)
    img = ImageTk.PhotoImage(image=im)
    panel = tk.Label(frame, text="Live Cam").pack()
    panel_image = tk.Label(frame, image=img).pack()
    after_id = frame.after(10, live_img)

def live_setup():
    global live_tab,after_id;
    if live_tab:
        load_splash()
        live_btn_text.set("Live")
        frame.after_cancel(after_id)
        live_tab = False
    else:
        live_btn_text.set("Stop")
        live_img()
        live_tab = True
        
    
def emotion(img):
    gray_img= cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces_detected = face_haar_cascade.detectMultiScale(gray_img, 1.32, 5)
    for (x,y,w,h) in faces_detected:
        cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),thickness=7)
        roi_gray=gray_img[y:y+w,x:x+h]#cropping region of interest i.e. face area from  image
        roi_gray=cv2.resize(roi_gray,(48,48))
        img_pixels = image.img_to_array(roi_gray)
        img_pixels = np.expand_dims(img_pixels, axis = 0)
        img_pixels /= 255
        predictions = model.predict(img_pixels)
        max_index = np.argmax(predictions[0])
        emotions = ('angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral')
        predicted_emotion = emotions[max_index]
        cv2.putText(img, predicted_emotion, (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
    resized_img = img
    return resized_img

def image_resize(im, inter = cv2.INTER_AREA):
    h, w, c = im.shape
    if w > h:
        r = 600 / w
    else:
        r = 480 / h
        
    width = int(im.shape[1] * r)
    height = int(im.shape[0] * r)
    dim = (width, height)
    resized = cv2.resize(im, dim)
    return resized


root = tk.Tk()
root.title('EMOTIONS')
root.iconbitmap("assets/ico.ico")
root.resizable(False, False)

canvas = tk.Canvas(root, height=480, width=600, bg='white')
canvas.pack()

frame = tk.Frame(root, bg='white')
frame.place(relwidth=1, relheight=1, relx=0, rely=0)

img_btn = tk.Button(root, text='Select Image', padx=35, pady=10, command=load_img)
img_btn.pack(side=tk.LEFT)
live_btn_text = tk.StringVar()
live_btn_text.set("Live")
live_btn = tk.Button(root, textvariable=live_btn_text, padx=35, pady=10,command=live_setup)
live_btn.pack(side=tk.RIGHT)

load_splash()
root.mainloop()
