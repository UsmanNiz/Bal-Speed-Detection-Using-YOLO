# Use the Ubuntu 18.04 base image
FROM nvidia/cuda:11.3.0-cudnn8-devel-ubuntu18.04

# Update package lists and install necessary dependencies
RUN apt-get update && \
    apt-get install -y wget bzip2 ca-certificates curl git

# Download and install Miniconda
RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh && \
    /bin/bash miniconda.sh -b -p /opt/conda && \
    rm miniconda.sh

# Add Miniconda to the PATH
ENV PATH="/opt/conda/bin:${PATH}"

# Create a new conda environment
RUN conda create -y --name myenv python=3.7

SHELL ["/bin/bash", "--login", "-c"]




RUN echo "conda activate myenv" >> ~/.bashrc

ADD "https://www.random.org/cgi-bin/randbyte?nbytes=10&format=h" skipcache
RUN git clone  https://github.com/UsmanNiz/Bal-Speed-Detection-Using-YOLO.git
WORKDIR "/Bal-Speed-Detection-Using-YOLO"
RUN mkdir weights
RUN curl -L -o weights/file.ext "https://drive.google.com/uc?export=download&id=1dYO0l-O_6T3A_uSnx5UGo7E0QQAJPDzh"

RUN source activate myenv && \
    pip install -r Bal-Speed-Detection-Using-YOLO/requirements.txt 
    #pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu113
# Activate the conda environment
#SHELL ["conda", "run", "-n", "fyp", "/bin/bash", "-c"]


# Install any necessary Python packages



# Set the default command to run detect.py
CMD ["python", "newui.py"]

