import argparse
import os
import numpy as np
from tqdm import tqdm
from src.dataset import  RetinalDataset
from src.model import get_torchvision_model
from src.transformation import inference_transformation
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

def scoring(gt, pred, thresh_hold = 0.5):
        gt = gt.to("cpu").numpy()
        pred = (pred.to("cpu").detach().numpy() > thresh_hold) * 1
        intersection = np.logical_and(pred, gt).sum()
        union = np.logical_or(pred, gt).sum()
        if union != 0:
            iou = intersection / union
        return iou

def epoch_evaluating( model, loss_criteria, device, val_loader, size, output_path):
        model.eval()
        ious = 0.0
        valid_loss = 0.0
        with torch.no_grad(): # Turn off gradient
            # For each batch
            for step, (images, labels) in tqdm(enumerate(val_loader)):
                # Transform X, Y to autogradient variables and move to device (GPU)
                fig = plt.figure(figsize=(10, 10))
                images = images.to(device, dtype=torch.float)
                labels = labels.to(device, dtype=torch.float)
                # Update groundtruth values
                outputs = model(images)
                iou = scoring(labels, outputs)
                
                #Plotting
                mask = outputs.reshape((size,size))
                mask = (mask.to("cpu").detach().numpy() > 0.5) * 1
                images = images.squeeze(0).permute(1,2,0)
                images = images.to("cpu")
                labels = labels.reshape((size, size))
                labels = labels.to("cpu")
                plt.imshow(images)
                plt.imshow(labels, cmap='copper', alpha=0.2)
                plt.imshow(mask, cmap='gray',alpha=0.15)
                plt.savefig(os.path.join(output_path, "OCTA_"+str(step)+".png"), dpi=fig.dpi, bbox_inches='tight')

                

#                 valid_loss += loss.item()
                ious+= iou
             # Clear memory
        del images, labels
        # return validation loss, and metric score
        print("valid IoU: " + str(ious / len(val_loader)))
        return  ious / len(val_loader)

def main():
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--path_images", help="path lead to folder which contains images or path to single image")
    parser.add_argument("--model_type", help= " model_type: efficentnet_b0, efficientnet_b1, efficientnet_b2, efficientnet_b3, efficientnet_b4, efficientnet_b5, Se_resnext50, Se_resnext101, Se_resnet50, se_resnet101, Se_resnet152, Resnet18, Resnet34,Resnet50, Resnet101", default= 'Se_resnext50')
    parser.add_argument("--weight", help=" path lead to model_weight")
    # parser.add_argument("")

    arg = parser.parse_args()
    
    #checking output folder
    if os.path.exists("./output"):
        pass
    else:
        os.mkdir("./output")
    output_path = os.path.dirname(os.path.realpath("./output"))
    output_path = os.path.join(output_path,"output")
    model = get_torchvision_model(arg.model_type, True, 1)

    #loading weight
    state_dict = torch.load(arg.weight)
    state_dict = state_dict["state"]
    model.load_state_dict(state_dict)

    #Loading dataset
    test_set = RetinalDataset(arg.path_images, 256, inference_transformation, phase = 'test')
    test_loader = torch.utils.data.DataLoader(test_set, batch_size =1, shuffle=False, num_workers=4)

    if os.path.isdir(arg.path_images):
        loss_criteria = nn.BCEWithLogitsLoss()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)
        ious = epoch_evaluating( model, loss_criteria, device, test_loader, 256, output_path)
    else:
        print("error input path_images")
if __name__ == '__main__':
    main()
