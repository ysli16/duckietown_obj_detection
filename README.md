## Overview 
This is for the object detection exercise of AMoD course 2020 at ETHz. The repo is based on the object-detection-ex-template provided by this link https://github.com/duckietown-ethz/object-detection-ex-template.git.

The repo is consisted of 3 parts.

### 1. Data collection
`data_collection` folder collects camera stream as well as the detected object bounding boxes and classes from the simulator. `draw_data.py` reads data from collected dataset, draws detection results(bounding boxes) on the original image and save it to `image` folder.

### 2. Model
`model` folder contains the model defination and interface for evaluation(`model.py`), the training script(`train.py`), trained model weights(`weights` folder) and local evaluation script and some testsets(`eval.py`, `*_testset`). The local evaluation script saves prediction results to `prediction` folder.

### 3. Evaluation
`eval` folder is used for evaluating the built docker image.

## How to use
### 1. Data collection
Edit following lines in `data_collection.py` (line 9-10)

        with makedirs("./<dataset>"):

                np.savez(f"./<dataset>/{npz_index}.npz", *(img, boxes, classes))
        
        
Replace `<dataset>` with dataset name that you want to use.

Then run command inside `data_collection` folder

`python3 data_collection.py`

The collection procedure will start and save collected data to `<dataset>` folder.

To visualize collected data, edit line 22 in `draw_data.py` , replace `<dataset>` with dataset name that you want to see.

        coll_file=os.path.join(cur_dir,'<dataset>',str(idx)+'.npz')

Then run command

`python3 draw_data.py`

The collected images with the bounding boxes drawn will be saved in `image` folder.

### 2. Model
To train the model, edit line 17 and line 21 in the `train.py`.

        self.data = list(sorted(os.listdir(os.path.join(root, 'data_collection','<dataset>'))))

        data_path = os.path.join(self.root, 'data_collection','<dataset>', self.data[idx])

Replace `<dataset>` with dataset name that you want to use for training. Make sure the training set is stored in `data_collection` folder.

To evaluate prediction result, edit line 64 in the `eval.py`.

        dataset_files = list(filter(lambda x: "npz" in x, os.listdir("./<testset>")))

Replace `<testset>` with dataset name that you want to use for evaluation. The testset is stored in `model` folder. 

### 3. Evaluation
First we need to build the docker image for trained model. To build the image, run command at the root folder

`docker build . -t <dockerid>/<imagename>`

Replace `<dockerid>` with your dockerhub username and `<imagename>` with name of this image.

To evaluate the image, switch to `eval` folder and run command

`make eval-gpu SUB=<dockerid>/<imagename>` for GPU evaluation

or

`make eval-cpu SUB=<dockerid>/<imagename>` for CPU evaluation

Edit `<dockerid>/<imagename>` as mentioned above.

*Tips for using GPU inside the container:    
The model image is based on pytorch/pytorch:1.7.0-cuda11.0-cudnn8-devel, which enables the container to use CUDA11.0.     
You also need to have NVIDIA Container Toolkit installed on the host computer before building the image.        
Refer to https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html#installing-on-ubuntu-and-debian for installation instruction.*
