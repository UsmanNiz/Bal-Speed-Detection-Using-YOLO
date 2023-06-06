FROM nvidia/cuda:11.3.0-cudnn8-devel-ubuntu18.04

RUN apt-get update && apt-get install -y \
    apache2 \
    curl \
    git \
    unzip\
    python3.7 \
    python3-pip



RUN git clone  https://github.com/UsmanNiz/Bal-Speed-Detection-Using-YOLO.git
RUN cd Bal-Speed-Detection-Using-YOLO
RUN mkdir weights
RUN curl -L -o /weights/file.ext "https://drive.google.com/uc?export=download&id=1dYO0l-O_6T3A_uSnx5UGo7E0QQAJPDzh"


RUN pip3 install -r Bal-Speed-Detection-Using-YOLO/requirements.txt  # install
RUN pip3 install torch==1.12.1+cu113 torchvision==0.13.1+cu113 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu113


# Commands to run Tkinter application
CMD ["Bal-Speed-Detection-Using-YOLO/newui.py"]
ENTRYPOINT ["python"]