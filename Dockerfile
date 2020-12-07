FROM pytorch/pytorch:1.7.0-cuda11.0-cudnn8-devel
RUN apt-get update && apt install libgl1-mesa-glx libglib2.0-0 -y
RUN pip install opencv-python numpy pillow tqdm
WORKDIR /workdir
COPY model/weights weights
COPY model/model.py model.py

RUN ls

# Don't forget to instantiate your model in case it needs to download a pretrained backend!
RUN python3 -c "from model import Wrapper; wrapper = Wrapper();"
