# syntax=docker/dockerfile:1
FROM ubuntu:20.04
WORKDIR /code
RUN apt update
RUN apt-get install -y \
    libpng-dev \
    freetype* \
    libblas-dev \
    liblapack-dev \
    libatlas-base-dev \
    gfortran
RUN apt-get install -y gcc musl-dev python3-pip libgl1
RUN apt-get install git -y
RUN git clone https://github.com/UsmanNiz/Bal-Speed-Detection-Using-YOLO.git  # clone
RUN cd Bal-Speed-Detection-Using-YOLO
RUN pip install -r requirements.txt  # install
RUN pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu113
ENTRYPOINT [ "python3" ]
CMD [ "newui.py" ]