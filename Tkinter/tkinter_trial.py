#### Root is the main window
#### Frame and canvas are elements of root. Frame lies in (0,0) and Canvas in (0,1)
#### Widgets lie in frame

import tkinter as tk
from PIL import ImageTk, Image

# Definition of important functions and class definitions

class create_Scale:
    def __init__(self, scale_loc, scale_name, scale_start, scale_end, scale_set_val):
        self.scale_loc =  scale_loc
        self.label = scale_name
        self.scale_start = scale_start
        self.scale_end = scale_end
        self.initial_val = scale_set_val

    def set_scale(self):
        scale = tk.Scale(self.scale_loc, from_=self.scale_start, to=self.scale_end, orient=tk.HORIZONTAL, label=self.label)
        scale.set(self.initial_val)
        scale.grid(row = 30, column = 15)
        return scale


def canvas_functions():
    if line.get() == 1:
        canvas.bind("<B1-Motion>", draw_line)
    if erase.get() ==1:
        canvas.bind("<B1-Motion>", erase_line)

def save_position(event):
    global last_x, last_y
    last_x, last_y = event.x, event.y

def erase_line(event):
    brush_size = s11.get()
    canvas.create_line(last_x, last_y, event.x, event.y, width=brush_size, capstyle='round', fill='white')
    save_position(event)

def draw_line(event):
    brush_size = s11.get()
    canvas.create_line(last_x,last_y, event.x, event.y, width = brush_size, capstyle = 'round', fill = 'blue')
    save_position(event)

def undo(event):

    pass

# Defining root window
root = tk.Tk()
root.geometry('2000x2000')


# Defining root configuration
root.columnconfigure(0, weight = 1)
root.columnconfigure(1,weight = 10)


# Defining Frame
frame = tk.Frame(root)
frame.grid(row = 0, column = 0, sticky = 'n')   # Frame configuration


#Defining Canvas
canvas = tk.Canvas(root, width = 1500, height = 1500)
canvas.grid(row = 0, column = 1, sticky = 'N')


# Defining and editing image
img_link = '/home/gen79net/Downloads/Image Detection/GUI_Files/runs/segment/predict/1e361b14-image85.jpg'
img_big = Image.open(img_link)
w,h = img_big.size
factor = 0.25    # Choose this factor to adjust image on canvas
img_resize = img_big.resize((int(w*factor),int(h*factor)))
img = ImageTk.PhotoImage(img_resize)
canvas.create_image( 80,0, anchor = tk.NW, image = img)


# Defining Scale
# s1 = tk.Scale(frame, from_ = 0, to = 50, orient = tk.HORIZONTAL, label = 'Brush Size')
# s1.set(10)
# s1.grid(row = 30, column = 15, sticky = 'w')

s1 = create_Scale(scale_loc = frame, scale_name = 'trial', scale_start = 0, scale_end = 100, scale_set_val = 10)
s11 = s1.set_scale()


# Adding Functions
## Checkbutton var definition
line = tk.IntVar()
erase = tk.IntVar()
## Checkbutton Definition
line_button = tk.Checkbutton(frame, text = 'Line', font = ('Arial', 12), variable = line, state = tk.ACTIVE,
                           onvalue = 1, offvalue = 0, command = lambda: erase.set(0))
erase_button = tk.Checkbutton(frame, text = 'Erase', font = ('Arial', 12), variable = erase,
                            onvalue = 1, offvalue = 0, command = lambda: line.set(0))
##Checkbutton positioning
line_button.grid(row = 8, column = 0)
erase_button.grid(row = 9, column = 0)

button = tk.Button(frame, text = 'Select', command = canvas_functions)
button.grid(row = 10, column = 0, rowspan = 2, columnspan = 2)

# Defining Canvas Motions
canvas.bind("<Button-1>", save_position)

root.mainloop()



