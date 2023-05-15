import tkinter as tk

class Notification(tk.Toplevel):
    def __init__(self, parent, text, timeout=5000):
        tk.Toplevel.__init__(self, parent)
        self.title('Speed Notification')
        #self.geometry('200x100+400+400')
        self.grid(row=3,column=0,padx=(20, 20),pady=(20,20), sticky="nsew")
        self.configure(background='white')
        self.attributes('-topmost', True)

        label = tk.Label(self, text=text)
        label.pack(pady=20)

        self.after(timeout, self.destroy)