import os
import sys
import copy
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from torch.autograd import Variable
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch import nn, optim
import torch
import torch.utils.data as utils
from src.loss import lovasz_softmax, binary_xloss, FocalLoss
from src.dataset import RetinalDataset
from src.metric import multi_label_f1
from src.transformation import inference_transformation, train_transformation
#loading model
from src.model import get_torchvision_model
##################

class Trainer(object):

    def __init__(self, **args):
        for key in args:
            setattr(self, key, args[key])

class FAZSegmentation(Trainer):

    def __init__(self, **args):
        super(FAZSegmentation, self).__init__(**args)

    def epoch_training(self, epoch, model,loss_criteria, device, train_loader, optimizer):
        model.train()
        training_loss = 0
        for batch, (inputs, labels) in enumerate(train_loader):
            inputs = inputs.to(device, dtype=torch.float)
            labels = labels.to(device, dtype=torch.float)
            optimizer.zero_grad()

            ##########feedforward############
            result = model(inputs)
            if self.loss.startswith("focal") or self.loss.startswith("smooth"):
                result = torch.sigmoid(result)
            loss = loss_criteria(result, labels)
            loss.backward()
            optimizer.step()
    
            training_loss += loss.item()
            sys.stdout.write(f"\rEpoch {epoch+1}... Training step {batch+1}/{len(train_loader)}")
        # Clear memory after training
        print("Training Loss: "+ str(training_loss/len(train_loader)))
#         del inputs, labels, loss
        # return training loss
        return training_loss/len(train_loader)

    def scoring(self, gt, pred):
        gt = gt.to("cpu").numpy()
        pred = (pred.to("cpu").detach().numpy() > self.thresh_hold) * 1
        intersection = np.logical_and(pred, gt).sum()
        union = np.logical_or(pred, gt).sum()
        if union != 0:
            iou = intersection / union
        return iou

    def epoch_evaluating(self, model, loss_criteria, device, val_loader):
        model.eval()
        ious = 0.0
        valid_loss = 0.0
        with torch.no_grad(): # Turn off gradient
            # For each batch
            for step, (images, labels) in enumerate(val_loader):

                # Transform X, Y to autogradient variables and move to device (GPU)
                images = images.to(device, dtype=torch.float)
                labels = labels.to(device, dtype=torch.float)

                # Update groundtruth values
                outputs = model(images)

                # calculating ious
                iou = self.scoring(labels, outputs)
                ious+= iou

             # Clear memory
        del images, labels
        # return validation loss, and metric score
        print("valid IoU: " + str(ious / len(val_loader)))
        return  ious / len(val_loader)

    def get_training_object(self):
        model = get_torchvision_model(self.net_type, self.pretrained, self.classes, self.loss)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)
        if self.weight_path is not None:
            state_dict = torch.load(
                self.weight_path)
            # state_dict = state_dict["state"]
            model.load_state_dict(state_dict)

        #### optimizer
        optimizer = torch.optim.Adam(model.parameters(), lr = self.lr, betas=(0.9, 0.999))
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode = "max", factor = self.factor, patience= self.patience)
#         model = nn.DataParallel(model)
          #  scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=10)

#         loss_criteria =torch.nn.BCELoss()
        if self.loss.startswith("focal"):
            loss_criteria = FocalLoss(gamma=self.gamma)
        else:
            loss_criteria = nn.BCEWithLogitsLoss()
        return model, loss_criteria, optimizer, scheduler

    def save_model(self, model):
        os.makedirs(self.model_path, exist_ok = True)
        model_path = os.path.join(self.model_path, f'{self.net_type}_model.pth')
        saveModule = model
#         saveModule = list(model.children())[0]
        model_specs = {"state": saveModule.state_dict(),
                       "batch_size": self.batch_size,
                       "lr": self.lr,
                       "size": self.size,
                       "net_type": self.net_type}
        torch.save(model_specs, model_path)
        return saveModule.state_dict()
    def train(self):
        best_score = 0

        # Init objects for training
        model, loss_criteria, optimizer, scheduler = self.get_training_object()

        #checking cuda
        if torch.cuda.is_available():
            print("training on GPU")
        else:
            print("training on CPU")
            
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        ##loading train_loader
        train_set = RetinalDataset(self.train_folder, self.size, train_transformation, phase = 'train')
        train_loader = torch.utils.data.DataLoader(train_set, batch_size = self.batch_size, shuffle=True, num_workers=1)

        ## loading validation_loader
        valid_set = RetinalDataset(self.valid_folder, self.size, inference_transformation, phase = 'valid')
        valid_loader = torch.utils.data.DataLoader(valid_set, batch_size = self.batch_size, shuffle=False, num_workers=4)

        ### training process
        for epoch in range(self.epoch):
            train_loss = self.epoch_training(epoch, model, loss_criteria, device, train_loader, optimizer)

            ## epoch evalulating
            new_score = self.epoch_evaluating(model, loss_criteria, device, valid_loader)
            scheduler.step(new_score)
            if best_score <= new_score:
                best_score = new_score
                state_dict = self.save_model(model)
#                 best_model_wts = copy.deepcopy(state_dict)
#         model.load_state_dict(best_model_wts)
        print("Best score: " + str(best_score))
        return model
