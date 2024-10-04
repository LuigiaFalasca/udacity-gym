import pandas as pd
import albumentations
import json
import PIL
from model.lane_keeping.dave.dave_model_v2 import Dave2
import statistics
import matplotlib.pyplot as plt
import pathlib
import torchvision
import numpy as np

def apply_augm(transf, img_array):
    match transf:
        case "CLAHE":
            augmented = albumentations.CLAHE(p=1)(image=img_array)
            return PIL.Image.fromarray(augmented['image'])
        case "Downscale":
            augmented = albumentations.Downscale(p=1)(image=img_array)
            return PIL.Image.fromarray(augmented['image'])
        case "ChannelDropout":
            augmented = albumentations.ChannelDropout(p=1)(image=img_array)
            return PIL.Image.fromarray(augmented['image'])
        case "HueSaturationValue":
            augmented = albumentations.HueSaturationValue(p=1)(image=img_array)
            return PIL.Image.fromarray(augmented['image'])
        case "Blur":
            augmented = albumentations.Blur(p=1)(image=img_array)
            return PIL.Image.fromarray(augmented['image'])
        case "RGBShift": 
            augmented = albumentations.RGBShift(p=1)(image=img_array)
            return PIL.Image.fromarray(augmented['image'])
        case "ChannelShuffle":
            augmented = albumentations.ChannelShuffle(p=1)(image=img_array)
            return PIL.Image.fromarray(augmented['image'])
        case "ToGray":
            augmented = albumentations.ToGray(p=1)(image=img_array)
            return PIL.Image.fromarray(augmented['image'])
        case "ColorJitter":
            augmented = albumentations.ColorJitter(p=1)(image=img_array)
            return PIL.Image.fromarray(augmented['image'])
        case "Defocus":
            augmented = albumentations.Defocus(p=1)(image=img_array)
            return PIL.Image.fromarray(augmented['image'])
        case "Equalize":
            augmented = albumentations.Equalize(p=1)(image=img_array)
            return PIL.Image.fromarray(augmented['image'])
        case "InvertImg":
            augmented = albumentations.InvertImg(p=1)(image=img_array)
            return PIL.Image.fromarray(augmented['image'])
        case "PlanckianJitter":
            augmented = albumentations.PlanckianJitter(p=1)(image=img_array)
            return PIL.Image.fromarray(augmented['image'])
        case "ZoomBlur":
            augmented = albumentations.ZoomBlur(p=1)(image=img_array)
            return PIL.Image.fromarray(augmented['image'])
        case "Sharpen":
            augmented = albumentations.Sharpen(p=1)(image=img_array)
            return PIL.Image.fromarray(augmented['image'])
        case "RingingOvershoot":
            augmented = albumentations.RingingOvershoot(p=1)(image=img_array)
            return PIL.Image.fromarray(augmented['image'])
        case "Superpixels":
            augmented = albumentations.Superpixels(p=1)(image=img_array)
            return PIL.Image.fromarray(augmented['image'])
        case "Spatter":
            augmented = albumentations.Spatter(p=1)(image=img_array)
            return PIL.Image.fromarray(augmented['image'])
        case "ToSepia":
            augmented = albumentations.ToSepia(p=1)(image=img_array)
            return PIL.Image.fromarray(augmented['image'])
        case "RandomFog":
            augmented = albumentations.RandomFog(p=1)(image=img_array)
            return PIL.Image.fromarray(augmented['image'])
        case "Solarize":
            augmented = albumentations.Solarize(p=1)(image=img_array)
            return PIL.Image.fromarray(augmented['image'])
        case "Posterize":
            augmented = albumentations.Posterize(p=1)(image=img_array)
            return PIL.Image.fromarray(augmented['image'])
        case "RandomBrightnessContrast":
            augmented = albumentations.RandomBrightnessContrast(p=1)(image=img_array)
            return PIL.Image.fromarray(augmented['image'])
        case "RandomGamma":
            augmented = albumentations.RandomGamma(p=1)(image=img_array)
            return PIL.Image.fromarray(augmented['image'])
        case "RandomSnow":
            augmented = albumentations.RandomSnow(p=1)(image=img_array)
            return PIL.Image.fromarray(augmented['image'])
        case "RandomRain":
            augmented = albumentations.RandomRain(p=1)(image=img_array)
            return PIL.Image.fromarray(augmented['image'])
        case _:
            return PIL.Image.fromarray(img_array)


def pred(transf):
    #get the nn
    checkpoint_path = pathlib.Path("~/work/code/udacity-gym/new_data/dave2-v2.ckpt")
    model = Dave2.load_from_checkpoint(checkpoint_path)
    model = model.to("cuda:0")
    model.eval()
    
    # Step 1: Prepare lists to store results for each column
    day_predictions = []
    daynight_predictions = []
    
    #get images from the folder day
    df = pd.read_csv("~/work/code/udacity-gym/new_data/udacity_dataset_lake_dave_05/lake_sunny_day/log.csv")

    #iteration to predict on day dataset
    for img_filename in df['image_filename']:
        img = PIL.Image.open("new_data/udacity_dataset_lake_dave_05/lake_sunny_day/image/" + img_filename)
        img_array = np.array(img)
        
        #augmentation prediction
        augmented = apply_augm(transf, img_array)
        input_image = torchvision.transforms.ToTensor()(augmented)
        input_image = input_image.to("cuda:0")
        day_predictions.append(model(input_image).item())
    
    
    #get images from the folder daynight
    df = pd.read_csv("~/work/code/udacity-gym/new_data/udacity_dataset_lake_dave_05/lake_sunny_daynight/log.csv")

    #iteration to predict on daynight dataset
    for img_filename in df['image_filename']:
        img = PIL.Image.open("new_data/udacity_dataset_lake_dave_05/lake_sunny_daynight/image/" + img_filename)
        img_array = np.array(img)
        
        #augmentation prediction
        augmented = apply_augm(transf, img_array)
        input_image = torchvision.transforms.ToTensor()(augmented)
        input_image = input_image.to("cuda:0")
        daynight_predictions.append(model(input_image).item())
    
    # Step 4: Create a DataFrame from the results
    df = pd.DataFrame({
        'Day': day_predictions,
        'DayNight': daynight_predictions,
    })
    
    # Step 5: Save the DataFrame to a CSV file
    df.to_csv('prediction_augm/new_data/'+ transf +'.csv', index=False)
    
    print("DataFrame saved to "+ transf+".csv")
    return


if __name__ == '__main__':  

    set_transformation = [ 
        "RandomFog",
        "RandomShadow"
        #"Defocus",  "ColorJitter", "Equalize", "InvertImg",
        #"PlanckianJitter", "ZoomBlur", "Sharpen", "RingingOvershoot"
    ]
    
    for transf in set_transformation:
        pred(transf)
    
    print("Experiment concluded")
