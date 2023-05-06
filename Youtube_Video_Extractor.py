import pytube
import tkinter as tk
from tkinter import messagebox
from tkinter.ttk import Progressbar
import threading
import subprocess

def download_video():
    def download():
        stream.download(filename="videos/yt_vid.mp4")
        progress_bar.destroy()
        subprocess.Popen(["python", "videoplayer.py"])
        root.destroy()

    def update_progress_bar(stream, chunk, bytes_remaining):
        total_size = stream.filesize
        bytes_downloaded = total_size - bytes_remaining
        progress = bytes_downloaded / total_size * 100
        progress_bar['value'] = progress

    url = video_url.get()
    try:
        video = pytube.YouTube(url, on_progress_callback=update_progress_bar)
        video.check_availability()
    except:
        messagebox.showerror("Error", "Please enter a valid YouTube video URL.")
        return

    stream = video.streams.filter(res="720p").first()
    if stream is None:
        stream = video.streams.filter(res="480p").first()
        if stream is None:
            stream = video.streams.get_highest_resolution()
    
    progress_bar = Progressbar(root, orient=tk.HORIZONTAL, length=200, mode='determinate')
    progress_bar.pack(pady=20)

    threading.Thread.daemon = True
    threading.Thread(target=download).start()
    
    stream.on_progress()
    progress_bar.start()

root = tk.Tk()
root.title("YouTube Video Extractor")
root.geometry("400x150")

video_url_label = tk.Label(root, text="Enter YouTube Video URL:")
video_url_label.pack(pady=5)
video_url = tk.Entry(root, width=50)
video_url.pack(pady=5)

go_button = tk.Button(root, text="Go", command=download_video)
go_button.pack(pady=5)

root.mainloop()
print('here')
# subprocess.Popen(["python", "videoplayer.py"])

print('here')
exit(0)