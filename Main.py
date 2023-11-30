import tkinter as tk
import tkinter as ttk
from PIL import ImageGrab
import torch
from torchvision.transforms import transforms
from Net import NetDropout
import torch.nn.functional as F
import os

class Decoder:
    def __init__(self, root):
        self.root = root
        self.settings()
        self.load_model()

        # Drawing
        self.canvas_frame = tk.Frame(root, width=200, height=300, bg='#808080', padx=self.padx)
        self.canvas_frame.pack(side=tk.LEFT)

        self.canvas = tk.Canvas(self.canvas_frame, bg="black", width=300, height=300, highlightthickness=0)
        self.canvas.pack()

        self.last_x, self.last_y = None, None

        self.canvas.bind("<B1-Motion>", self.draw)
        self.canvas.bind("<ButtonRelease-1>", self.reset_coordinates)

        self.clear_button = ttk.Button(text='Очистить', command=self.clear_canvas)
        self.clear_button.place(relx=0.5, rely=0.9, anchor="c", relwidth=0.7, relheight=0.07)

        # Proba display 
        self.proba_predict = tk.Frame(root, bg='white')
        self.current_image = None
        self.proba_predict.place(relx=0.65, rely=0.1, anchor="c", relwidth=0.33, relheight=0.1)
        self.prediction_var = tk.StringVar()
        self.prediction_var.set("")
        self.prediction_label = tk.Label(self.proba_predict, textvariable=self.prediction_var, font=('Helvetica', 25), bg='white', fg='Black')
        self.prediction_label.pack()

        # Predict number
        self.predict_number = tk.Frame(master=root, bg='white')
        self.predict_number.place(relx=0.3, rely=0.1, anchor="c", relwidth=0.15, relheight=0.1)
        self.prediction_number = tk.StringVar()
        self.prediction_number.set("")
        self.prediction_number_label = tk.Label(self.predict_number, textvariable=self.prediction_number, font=('Helvetica', 25), bg='white', fg='Black')
        self.prediction_number_label.pack()

    #Function 

    def load_model(self):
        self.model = NetDropout()
        self.model.load_state_dict(torch.load('weights/Model_50.pt'))
        self.model.eval()
    
    def settings(self):
        self.root.title("MNIST decoder")
        self.root.configure(bg='#808080')
        self.root.geometry("380x440")
        self.root.resizable(0, 0)
        self.padx = 50

    def update_prediction(self):
        if self.current_image is not None:
            processed_image  = self.preprocess(self.current_image)
            image_tensor = processed_image.unsqueeze(0)

            with torch.no_grad():
                
                output = self.model(image_tensor)
                _, label = torch.max(output, dim=1)
                proba = round(F.softmax(output, dim=1)[0, label].item() * 100, 2)
                
        self.prediction_var.set(f"{str(proba)} %")
        self.prediction_number.set(str(label.item()))

    def preprocess(self, image):
        transformer = transforms.Compose([
            transforms.Grayscale(num_output_channels=1),
            transforms.Resize(size=(28, 28)),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
        
        return transformer(image)

    def draw(self, event):
        if self.last_x and self.last_y:
            x, y = event.x, event.y
            self.canvas.create_line(self.last_x, self.last_y, x, y, fill="white", width=10)
            self.last_x = x
            self.last_y = y

            # Обновление текущего изображения
            x0 = self.root.winfo_rootx() + self.canvas_frame.winfo_x() + self.padx
            y0 = self.root.winfo_rooty() + self.canvas_frame.winfo_y()
            x1 = x0 + self.canvas.winfo_reqwidth()
            y1 = y0 + self.canvas.winfo_reqheight()
            
            self.current_image = ImageGrab.grab(bbox=(x0, y0, x1, y1))       
            self.update_prediction()
        else:
            self.last_x = event.x
            self.last_y = event.y

    def reset_coordinates(self, event):
        self.last_x, self.last_y = None, None

    def clear_canvas(self):
        self.canvas.delete("all")
        self.prediction_number.set("")
        self.prediction_var.set("")

if __name__ == "__main__":
    root = tk.Tk()
    app = Decoder(root)
    root.mainloop()