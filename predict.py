import argparse
import json
import torch
import futils
import numpy as np

ap = argparse.ArgumentParser(
    description='predict-file')
ap.add_argument('input_img', default='flowers/test/1/image_06743.jpg', nargs='*', action="store", type = str)
ap.add_argument('--top_k', default=5, dest="top_k", action="store", type=int)
ap.add_argument('--category_names', dest="category_names", action="store", default='cat_to_name.json')
ap.add_argument('--gpu', default="gpu", action="store", dest="gpu")

pa = ap.parse_args()
path_image = pa.input_img
number_of_outputs = pa.top_k
power = pa.gpu
input_img = pa.input_img


checkpoint_dict = torch.load('checkpoint.pth')
model,_,_ = futils.setup()
model.class_to_idx = checkpoint_dict['class_to_idx']
model.load_state_dict(checkpoint_dict['state_dict'])

with open('cat_to_name.json', 'r') as json_file:
    cat_to_name = json.load(json_file)


probabilities = futils.predict(path_image, model, number_of_outputs, power)


labels = [cat_to_name[str(index + 1)] for index in np.array(probabilities[1][0])]
probability = np.array(probabilities[0][0])


i=0
while i < number_of_outputs:
    print("{} with a probability of {}".format(labels[i], probability[i]))
    i += 1

print("Here you are")