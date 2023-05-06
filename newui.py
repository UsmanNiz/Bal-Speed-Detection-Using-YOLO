import tkinter
import tkinter.messagebox
import customtkinter
from PIL import Image,ImageTk
import os
import tkinter as tk
from tkinter import filedialog
import cv2
import time
from detect import *
from Youtube_Video_Extractor import run_yt_downloader

customtkinter.set_appearance_mode("System")  # Modes: "System" (standard), "Dark", "Light"
customtkinter.set_default_color_theme("blue")  # Themes: "blue" (standard), "green", "dark-blue"


class App(customtkinter.CTk):
    def __init__(self):
        super().__init__()

        # configure window
        self.title("The Diamond Cutter : 19k-0292 19k-0181 19k-1449")
        self.geometry(f"{1100}x{580}")

        # configure grid layout (4x4)
        self.grid_columnconfigure(1, weight=1)
        self.grid_columnconfigure((2, 3), weight=0)
        self.grid_rowconfigure((0, 1, 2), weight=1)
        self.current_file_path=""

        # create sidebar frame with widgets
        self.sidebar_frame = customtkinter.CTkFrame(self, width=140, corner_radius=0)
        self.sidebar_frame.grid(row=0, column=0, rowspan=4, sticky="nsew")
        self.sidebar_frame.grid_rowconfigure(4, weight=1)
        self.logo_label = customtkinter.CTkLabel(self.sidebar_frame, text="Diamond Cutter", font=customtkinter.CTkFont(size=20, weight="bold"))
        self.logo_label.grid(row=0, column=0, padx=20, pady=(20, 10))
        self.sidebar_button_1 = customtkinter.CTkButton(self.sidebar_frame, command=self.home_button_event,text="Home")
        self.sidebar_button_1.grid(row=1, column=0, padx=20, pady=10)
        self.sidebar_button_2 = customtkinter.CTkButton(self.sidebar_frame, command=self.URL_button_event,text="Stream URL")
        self.sidebar_button_2.grid(row=2, column=0, padx=20, pady=10)
        self.sidebar_button_3 = customtkinter.CTkButton(self.sidebar_frame, command=self.homebrew_button_event,text="Homebrew Videos")
        self.sidebar_button_3.grid(row=3, column=0, padx=20, pady=10)
        self.appearance_mode_label = customtkinter.CTkLabel(self.sidebar_frame, text="Appearance Mode:", anchor="w")
        self.appearance_mode_label.grid(row=5, column=0, padx=20, pady=(10, 0))
        self.appearance_mode_optionemenu = customtkinter.CTkOptionMenu(self.sidebar_frame, values=["Light", "Dark", "System"],
                                                                       command=self.change_appearance_mode_event)
        self.appearance_mode_optionemenu.grid(row=6, column=0, padx=20, pady=(10, 10))
        self.scaling_label = customtkinter.CTkLabel(self.sidebar_frame, text="UI Scaling:", anchor="w")
        self.scaling_label.grid(row=7, column=0, padx=20, pady=(10, 0))
        self.scaling_optionemenu = customtkinter.CTkOptionMenu(self.sidebar_frame, values=["80%", "90%", "100%", "110%", "120%"],
                                                               command=self.change_scaling_event)
        self.scaling_optionemenu.grid(row=8, column=0, padx=20, pady=(10, 20))

    
        

        # create textbox
        self.textbox = customtkinter.CTkTextbox(self, width=5)
        self.textbox.configure(wrap="word")
        self.textbox.grid(row=2,column=1, columnspan=2,padx=(20, 20),pady=(20,20), sticky="nsew")

        #main image
        image_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "UiImages")
        self.home_page_image = customtkinter.CTkImage(Image.open(os.path.join(image_path, "home_page_image.png")), size=(755, 550))
        self.home_page_image_label = customtkinter.CTkLabel(self, width=5,text="", image=self.home_page_image)
        self.home_page_image_label.grid(row=0, column=1,columnspan=3,padx=(20,20),pady=(20,0))
     

     
       
       
        self.appearance_mode_optionemenu.set("Dark")
        self.scaling_optionemenu.set("100%")
        
        self.textbox.insert("0.0", "The Diamond Cutter is a unique approach to finding cricket ball speed using modern AI techniques such as Action Recognition and Object Detection. The project runs on pre-recorded videos from your harddrive or directly from a video url from a hosting service. Both modes of functioning are added in this program. This Program runs on a GTX/RTX enabled machine with CUDA runtime and it's relevant pytorch distribution installed.CPU only mode to be added in the future. Before runnning this ensure you have the relevant python version installed and have installed the pre-trained weights.")

    def open_input_dialog_event(self):
        dialog = customtkinter.CTkInputDialog(text="Type in a number:", title="CTkInputDialog")
        print("CTkInputDialog:", dialog.get_input())

    def change_appearance_mode_event(self, new_appearance_mode: str):
        customtkinter.set_appearance_mode(new_appearance_mode)

    def change_scaling_event(self, new_scaling: str):
        new_scaling_float = int(new_scaling.replace("%", "")) / 100
        customtkinter.set_widget_scaling(new_scaling_float)

    def home_button_event(self):
         self.inference_label.grid_remove()
         self.home_page_image_label.grid(row=0, column=1,columnspan=3,padx=(20,20),pady=(20,0))
         self.textbox.grid(row=2,column=1, columnspan=2,padx=(20, 20),pady=(20,20), sticky="nsew")

    def URL_button_event(self):
        run_yt_downloader()
        
        self.current_file_path='videos/yt_vid.mp4'
        run_detector(self)

    def homebrew_button_event(self):
        self.current_file_path = filedialog.askopenfilename()
        if self.current_file_path:
        # Do something with the selected file path
            print(f"Selected file: {self.current_file_path}")
            print("sidebar_button click")
            self.textbox.grid_remove()
            self.home_page_image_label.grid_remove()
            self.inference_label=customtkinter.CTkLabel(self,width=850,height=550,text="")
            self.inference_label.grid(row=0, column=1,columnspan=3,padx=(20,20),pady=(20,20))
            run_detector(self)

  



        #     while cap.isOpened():
        #             ret, frame = cap.read()
        #             if ret:
        #                 frame = cv2.resize(frame, (750, 550))
        #                 img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        #                 img = Image.fromarray(img)
        #                 img = ImageTk.PhotoImage(img)
        # # Update the label with the new image
        #                 self.inference_label.configure(image=img)
        #                 self.inference_label.image = img
        # # Update the custom Tkinter window to show the new frame
        #                 self.update()
        # # Wait for a key event for 10 milliseconds
        #                 key = cv2.waitKey(10)
        #     cap.release()
        #     cv2.destroyAllWindows()
                

        


if __name__ == "__main__":
    app = App()
    app.mainloop()
