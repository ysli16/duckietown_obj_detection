import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, Subset
import torchvision
from PIL import Image
from engine import train_one_epoch
import model
import utils

class DTDataset(Dataset):
    def __init__(self, root):
        self.root=root #work dir: /home/yueshan/Desktop/AMoD/RH8/obj_detection/object-detection-ex-template
        
        # load all image files, sorting them to
        # ensure that they are aligned
        self.data = list(sorted(os.listdir(os.path.join(root, 'data_collection','dataset'))))

    def __getitem__(self, idx):
        # load data
        data_path = os.path.join(self.root, 'data_collection','dataset', self.data[idx])
        data=np.load(data_path)
    
        img = data[f"arr_{0}"]
        boxes = data[f"arr_{1}"]
        classes = data[f"arr_{2}"]
        
        num_objs=boxes.shape[0]
        
        # convert everything into a torch.Tensor
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(classes, dtype=torch.int64)
        image_id = torch.tensor([idx])
        
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        
        # suppose all instances are not crowd
        iscrowd = torch.zeros((num_objs,), dtype=torch.int64)

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = iscrowd
        
        img=Image.fromarray(np.uint8(img))# convert np array to PIL Image
        
        return img, target

    def __len__(self):
        return len(self.data)

def main():
    # TODO train loop here!
    # TODO don't forget to save the model's weights inside of `./weights`!
    # train on the GPU or on the CPU, if a GPU is not available
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    #device=torch.device('cpu')
    # use our dataset and defined transformations
    #root=os.path.dirname(os.getcwd())
    root=os.getcwd()
    dataset = DTDataset(root)

    # split the dataset in train and test set
    dataset_size=len(dataset)
    dataset_size=100
    indices = torch.randperm(dataset_size).tolist()
    dataset = Subset(dataset, indices[:-10])
    
    # define training and validation data loaders
    data_loader = DataLoader(
        dataset, batch_size=5, shuffle=True, num_workers=1,collate_fn=utils.collate_fn)

    # get the model using our helper function
    md = model.Model()

    # move model to the right device
    md.to(device)

    # construct an optimizer
    params = [p for p in md.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.005,
                                momentum=0.9, weight_decay=0.0005)
    # and a learning rate scheduler
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                   step_size=3,
                                                   gamma=0.1)

    # let's train it for 10 epochs
    num_epochs = 10

    for epoch in range(num_epochs):
        # train for one epoch, printing every 10 iterations
        train_one_epoch(md, optimizer, data_loader, device, epoch, print_freq=10)
        # update the learning rate
        lr_scheduler.step()
        # evaluate on the test dataset
        #evaluate(model, data_loader_test, device=device)

    print("That's it!")

if __name__ == "__main__":
    main()