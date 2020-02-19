# FAZ-Segmentation
Combine Hessian Filter and UNet for Foveal Avascular Zone Extraction

![image](https://drive.google.com/uc?export=view&id=1vPlOGi1sBtjhKNpXrwNoM7U5wR7ffcew)
# Docker Installation for FLask App


1. Build and run docker on port 2001
```
$ ./docker-build.sh

```
If getting error in permission
```
$ chmod u+x ./docker-build.sh

```
2. I have already mounted /AWS-web-app/app/static/images to /images of docker, so to test we will prepare image in:
```
├   ├── images
|       ├── raw
|           ├── 1.tif
|       ├── predict
|           ├── 1.png
```
*  We will put raw image in /images/raw
*  In postman: 
    * url: http://localhost:2001/faz/predict
    * METHOD: GET
    * Params: 
        * Key: id
        * value: name of image such as 1.tif
* You will have the prediction of model at /images/predict

# Training process

## Prepare dataset folder
```
├── train
|   ├── raw 
|       ├── image1.tif
|       ├── ...
|   ├── mask
|       ├── image1.png
|       ├── ...
├── valid
|   ├── raw 
|       ├── image1.png
|       ├── ...
|   ├── mask
|       ├── image1.png
|       ├── ...
├── test
|   ├── raw 
|       ├── image1.tif
|       ├── ...
|   ├── mask
|       ├── image1.png
|       ├── ...
```

## Setup Environment
Run this script to create a virtual environment and install dependency libraries
1.  $conda create -n name_environment python=3.6
2.  $conda activate name_environment
3.  $pip install -r requirements-2.txt

To train this project, we just run the command
```
$python train.py
```
where train_config.json which is located in config folder

We need to adjust the parameter in this json file before training:


*  net_type: name of pretrained model you want to train. 
list of model:
efficentnet_b0, efficientnet_b1, efficientnet_b2, efficientnet_b3, efficientnet_b4, efficientnet_b5, Se_resnext50, Se_resnext101, Se_resnet50, se_resnet101, Se_resnet152, Resnet18, Resnet34,Resnet50, Resnet101

*  pretrained: boolean, using pretrained weights from ImageNet

*  weight_path: Weight path of old trained model

*  train_folder : path of raw folder of training dataset
example: /home/vinhng/OCTA/preprocess_OCTA/train/raw

* valid_folder : path of raw folder of valid dataset
example: /home/vinhng/OCTA/preprocess_OCTA/valid/raw

* test_folder : path of raw folder of valid dataset
example: /home/vinhng/OCTA/preprocess_OCTA/test/raw

* classes: number of classes. Default = 1

* model_path: directory which contains trained model 

* size: size of input image and mask

* thresh_hold: thresh hold for convert grayscale mask to binary mask

* epoch: number of training epoch

# Testing process
```
download weight of model: 
https://storage.googleapis.com/v-project/Se_resnext50-920eef84.pth

Then move this weight in folder: 
./models
```

```
python test.py --path_images --model_type --weight 
```
* path_images: directory of raw folder in testset (see prepare dataset above)
* model_type: name of pretrained model you want to train. Default: Se_resnext50

List of pretrained model is at training process above
* weight: directory to weight path.
