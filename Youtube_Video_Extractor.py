import pytube
import tkinter as tk
from tkinter import messagebox
import threading
import customtkinter

class Extractor_Window(customtkinter.CTkToplevel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.geometry("400x250")
        self.title("Youtube Video Extractor")
        self.wm_attributes("")
        
        self.label = customtkinter.CTkLabel(self, text="Enter YouTube Video URL:")
        self.label.pack(padx=5, pady=20)

        self.textbox = customtkinter.CTkEntry(master=self, placeholder_text="Youtube URL", width=360)
        self.textbox.pack(padx=5)

        self.button = customtkinter.CTkButton(master=self, text="Go", command=self.download_video)
        self.button.pack(padx=5, pady=15)

        self.progresstext = customtkinter.CTkLabel(self, text="Download Progress:")
        self.progressbar = customtkinter.CTkProgressBar(master=self, orientation="horizontal")
        self.progressbar.set(0)

    def download_video(self):
        def download():
            stream.download(filename="videos/yt_vid.mp4")
            self.destroy()

        def update_progress_bar(stream, chunk, bytes_remaining):
            total_size = stream.filesize
            bytes_downloaded = total_size - bytes_remaining
            progress = bytes_downloaded / total_size
            self.progressbar.set(progress)
        
        self.button.configure(state="disabled")
        url = self.textbox.get()
        try:
            video = pytube.YouTube(url, use_oauth=True, allow_oauth_cache=True, on_progress_callback=update_progress_bar)
            video.check_availability()
        except:
            messagebox.showerror("Error", "Please enter a valid YouTube video URL.")
            self.button.configure(state="normal")
            return

        try:
            stream = video.streams.filter(res="720p").first()
            if stream is None:
                stream = video.streams.filter(res="480p").first()
                if stream is None:
                    stream = video.streams.get_highest_resolution()
        except:
            messagebox.showerror("Error", "URL is valid but video is unaccessable from Youtube's side.")
            self.button.configure(state="normal")
            return

        self.progresstext.pack(padx=5, pady=10)
        self.progressbar.pack(padx=5)

        self.attributes('-disabled', True)
        
        threading.Thread.daemon = True
        download_thread = threading.Thread(target=download)
        download_thread.start()

        stream.on_progress()


# def run_yt_downloader():
#     root = tk.Tk()
#     root.title("YouTube Video Extractor")
#     root.geometry("400x150")

#     video_url_label = tk.Label(root, text="Enter YouTube Video URL:")
#     video_url_label.pack(pady=5)
#     video_url = tk.Entry(root, width=50)
#     video_url.pack(pady=5)
#     go_button = tk.Button(root, text="Go", command=lambda:download_video(root,video_url))
#     go_button.pack(pady=5)
#     root.mainloop()
#     print('here')
#     print('here')