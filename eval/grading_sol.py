
from map_sol import mean_average_precision
from torchvision.transforms.functional import to_tensor
import os
import numpy as np
from model import Wrapper

dataset_files = list(filter(lambda x: "npz" in x, os.listdir("./dataset")))

true_boxes = []
pred_boxes = []

def make_boxes(id, labels, scores, bboxes):
    temp = []
    for i in range(len(labels)):
        x1 = bboxes[i][0]
        y1 = bboxes[i][1]
        x2 = bboxes[i][2] - x1
        y2 = bboxes[i][3] - y1

        temp.append([id, labels[i], scores[i], x1, y1, x2, y2])
    return temp

wrapper = Wrapper()

BATCH_SIZE = 2
BATCH_QTTY = int(len(dataset_files) / BATCH_SIZE)

def make_batches(list):
    for i in range(0, len(list), BATCH_SIZE):
        yield list[i:i+BATCH_SIZE]

from tqdm import trange
batches = list(make_batches(dataset_files[:BATCH_QTTY*BATCH_SIZE]))
for nb_batch in trange(len(batches)):
    batch = batches[nb_batch]

    for nb_img, file in enumerate(batch):
        with np.load(f'./dataset/{file}') as data:
            img, boxes, classes = tuple([data[f"arr_{i}"] for i in range(3)])

            p_boxes, p_classes, p_scores = wrapper.predict(np.array([img]))

            for j in range(len(p_boxes)):
                pred_boxes += make_boxes(nb_batch+nb_img, p_classes[j], p_scores[j], p_boxes[j])
            true_boxes += make_boxes(nb_batch+nb_img, classes, [1.0]*len(classes), boxes)

true_boxes = np.array(true_boxes, dtype=float)
pred_boxes = np.array(pred_boxes, dtype=float)
# print(mean_average_precision(pred_boxes, true_boxes, box_format="midpoint", num_classes=5))
print(mean_average_precision(pred_boxes, true_boxes, box_format="midpoint", num_classes=5).item())
# approx 87%!!!
