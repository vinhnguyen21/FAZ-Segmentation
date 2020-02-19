import os 
import torch
import numpy as np
np.seterr(divide='ignore', invalid='ignore')
import cv2
import matplotlib.pyplot as plt
from PIL import Image

from .model import get_torchvision_model
from .transformation import tensor_transform
from .dataset import FAZ_Preprocess
from torch.utils.model_zoo import load_url

# model_url
url = "https://storage.googleapis.com/v-project/Se_resnext50-920eef84.pth"

class FAZ_Classifier():
    def __init__(self, model_type = "Se_resnext50", weight_path = None):

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        if model_type:
            self.model = get_torchvision_model(model_type, True, 1, None)

        if weight_path is not None:
            state_dict = torch.load(weight_path, map_location = self.device)
        else:
            state_dict = load_url(url, map_location = self.device)
            state_dict = state_dict["state"]
        
        #loading weight to model
        self.model.load_state_dict(state_dict)
        self.size = 256

    def transform(self, image):
        return tensor_transform(image, self.size)

    def predict(self, image_path, prediction_folder):
        
        #image name
        image_name = image_path.split("/")[-1]

        #checking extension
        ext = image_name.rsplit(".")[1]
        if not ext == "png":
            image_name = image_name.rsplit('.')[0] + '.png'

        print(prediction_folder)
        # enhance image with Hessian Filter
        image = FAZ_Preprocess(image_path, [0.5,1, 1.5, 2, 2.5],1, 2)
        image = image.vesselness2d()
        image = Image.fromarray(image.astype(np.float32)*255).convert('RGB')
        image = self.transform(image)
        
        #prediction
        mask = image.unsqueeze(0)
        self.model.eval()
        mask = self.model(mask)
        mask = (mask.to("cpu").detach().numpy() > 0.6) * 1
        mask = mask.reshape(self.size, self.size)

        image = image.permute(1,2,0).numpy()
        # image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

        #overlay original image with mask
        test = mask.copy()
        test = mask.astype(np.uint8)
        contours, hierarchy = cv2.findContours(test,cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        img = image.copy()
        img = img.astype(np.uint8)
        a = cv2.drawContours(img, contours,-1,(255, 255,0),1)

        #setup figure without axis
        fig = plt.figure(frameon=False)
        ax = plt.Axes(fig, [0., 0., 1., 1.])
        ax.set_axis_off()
        fig.add_axes(ax)
        ax.imshow(image, cmap='gray')
        ax.imshow(a, alpha=0.3)
        plt.savefig(os.path.join(prediction_folder, image_name), bbox_inches='tight', pad_inches=0)

        return os.path.join(prediction_folder, image_name)