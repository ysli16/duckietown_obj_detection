import torch
import torchvision
from torchvision.transforms.functional import to_tensor
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

class Wrapper:
    def __init__(self):
        # TODO Instantiate your model and other class instances here!
        # TODO Don't forget to set your model in evaluation/testing/production mode, and sending it to the GPU (if you have one)

        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.model = Model()
        self.model.load_state_dict(torch.load("weights/model.pt", map_location=self.device))
        self.model.to(self.device)
        self.model.eval()

    def predict(self, batch_or_image):
        # TODO: Make your model predict here!

        # The given batch_or_image parameter will be a numpy array (ie either a 224 x 224 x 3 image, or a
        # batch_size x 224 x 224 x 3 batch of images)
        # These images will be 224 x 224, but feel free to have any model, so long as it can handle these
        # dimensions. You could resize the images before predicting, make your model dimension-agnostic somehow,
        # etc.

        # This method should return a tuple of three lists of numpy arrays. The first is the bounding boxes, the
        # second is the corresponding labels, the third is the scores (the probabilities)

        if len(batch_or_image.shape) == 3:
            batch = [batch_or_image]
        else:
            batch = batch_or_image

        with torch.no_grad():
            preds = self.model([to_tensor(img).to(device=self.device, dtype=torch.float) for img in batch])

        boxes = []
        labels = []
        scores = []
        for pred in preds:
            boxes.append(pred["boxes"].cpu().numpy())
            labels.append(pred["labels"].cpu().numpy())
            scores.append(pred["scores"].cpu().numpy())

        return boxes, labels, scores


class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        # TODO Instantiate your weights etc here!
        model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)

        # get number of input features for the classifier
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        # replace the pre-trained head with a new one
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, 5)
        
        self.model=model

    def forward(self, x, y=None):
        return self.model(x) if y is None else self.model(x, y)
