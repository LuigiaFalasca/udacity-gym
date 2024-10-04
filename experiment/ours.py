import albumentations
import json
import pandas
import PIL
import pandas
from model.lane_keeping.dave.dave_model_v2 import Dave2
import statistics
import matplotlib.pyplot as plt
import pathlib
import torchvision
import numpy as np
import itertools

def apply_augm(transf, img_array):
    match transf:
        case "CLAHE":
            augmented8 = albumentations.CLAHE(p=1)(image=img_array)
            return PIL.Image.fromarray(augmented8['image'])
        case "Downscale":
            augmented7 = albumentations.Downscale(p=1)(image=img_array)
            return PIL.Image.fromarray(augmented7['image'])
        case "ChannelDropout":
            augmented6 = albumentations.ChannelDropout(p=1)(image=img_array)
            return PIL.Image.fromarray(augmented6['image'])
        case "HueSaturationValue":
            augmented5 = albumentations.HueSaturationValue(p=1)(image=img_array)
            return PIL.Image.fromarray(augmented5['image'])
        case "Blur":
            augmented4 = albumentations.Blur(p=1)(image=img_array)
            return PIL.Image.fromarray(augmented4['image'])
        case "RGBShift": 
            augmented3 = albumentations.RGBShift(p=1)(image=img_array)
            return PIL.Image.fromarray(augmented3['image'])
        case "ChannelShuffle":
            augmented2 = albumentations.ChannelShuffle(p=1)(image=img_array)
            return PIL.Image.fromarray(augmented2['image'])
        case "ToGray":
            augmented = albumentations.ToGray(p=1)(image=img_array)
            return PIL.Image.fromarray(augmented['image'])
        case _:
            return PIL.Image.fromarray(img_array)

if __name__ == '__main__':
    
        
    #get the nn
    checkpoint_path = pathlib.Path("~/work/code/udacity-gym/new_data/dave2-v2.ckpt")
    model = Dave2.load_from_checkpoint(checkpoint_path)
    model.eval()
    
    set_transformation = ["ToGray",  "ChannelShuffle", "RGBShift", "Blur",
                          "HueSaturationValue", "ChannelDropout", "Downscale", "CLAHE"]
    # Generate all combinations of 5 elements
    combinations = list(itertools.combinations(set_transformation, 5))
    
    #get collision day
    with open("new_data/udacity_dataset_lake_dave_05/lake_sunny_day/info.json") as f:
        data = json.load(f)
    
    collisions = []
    for i in range(1, len(data['events'])):
        collisions.append(data['events'][i]['timestamp'])
    
    #timestamp before the collision
    index_day= []
    timestamps_day=[]

    df = pandas.read_csv("~/work/code/udacity-gym/new_data/udacity_dataset_lake_dave_05/lake_sunny_day/log.csv")
    j=0
    for i in range(len(df['time'])-1):
        if j < len(collisions):
            if (int(collisions[j])>= int(df['time'][i]) and int(collisions[j])<= int(df['time'][i+1])):
                timestamps_day.append(
                     {'collision_tmp': collisions[j],
                      'before_tmp': df['time'][i],
                      'after_tmp': df['time'][i+1],
                      'before_index': i,
                      'after_index': i+1
                     }
                 )
                j= j+1
                index_day.append(i)
    
    #get collision daynight
    with open("new_data/udacity_dataset_lake_dave_05/lake_sunny_daynight/info.json") as f:
        data = json.load(f)
    
    collisions = []
    for i in range(1, len(data['events'])):
        collisions.append(data['events'][i]['timestamp'])
    
    #timestamp before the collision
    index_daynight= []
    timestamps_daynight=[]

    df = pandas.read_csv("~/work/code/udacity-gym/new_data/udacity_dataset_lake_dave_05/lake_sunny_daynight/log.csv")
    j=0
    for i in range(len(df['time'])-1):
        if j < len(collisions):
            if (int(collisions[j])>= int(df['time'][i]) and int(collisions[j])<= int(df['time'][i+1])):
                timestamps_daynight.append(
                     {'collision_tmp': collisions[j],
                      'before_tmp': df['time'][i],
                      'after_tmp': df['time'][i+1],
                      'before_index': i,
                      'after_index': i+1
                     }
                 )
                j= j+1
                index_daynight.append(i)
    
    
    k=1
    # Prediction using the combinations
    for combo in combinations:
        print(str(k), str(combo))
        variance_value = []
        #get images from the folder
        df = pandas.read_csv("~/work/code/udacity-gym/new_data/udacity_dataset_lake_dave_05/lake_sunny_day/log.csv")
        for img_filename in df['image_filename']:
            img = PIL.Image.open("new_data/udacity_dataset_lake_dave_05/lake_sunny_day/image/" + img_filename)
            img_array = np.array(img)
            predictions = []
            for transf in combo:
                augmented = apply_augm(transf, img_array)
                input_image = torchvision.transforms.ToTensor()(augmented)
                predictions.append(model(input_image).item())
            #variance
            variance_value.append(statistics.variance(predictions))
        
        
        if k<10:
            num_serie = "00" + str(k)
        elif k <100: 
            num_serie = "0" + str(k)
        else:
            num_serie = str(k)
            
        T = sorted(variance_value)[int(len(variance_value) * 0.95)]
        
        max_y = max(variance_value)
        chunk_size = 500
        for i in range(0, len(variance_value), chunk_size):
            xpoints = np.arange(i, i+ chunk_size)
            ypoints = np.array(variance_value[i: i+chunk_size])
            x0 = [num for num in index_day if (num >= i and num<=(i+chunk_size))]
            y0 = [variance_value[num] for num in x0]
            plt.rcParams["figure.figsize"] = (10,6)
            plt.plot(xpoints, ypoints)
            plt.plot(x0, y0, "o")
            plt.axhline(y=T, color='r', linestyle='-')
            plt.ylim(0, max_y)
            plt.title(" daynight - " + str(combo))
            plt.savefig("plot/ours/new_data/day/" + num_serie +"_plot_" +str(i)+".png")
            plt.clf()
            
            
        #daynight
        variance_value = []
        #get images from the folder
        df = pandas.read_csv("~/work/code/udacity-gym/new_data/udacity_dataset_lake_dave_05/lake_sunny_daynight/log.csv")
        for img_filename in df['image_filename']:
            img = PIL.Image.open("new_data/udacity_dataset_lake_dave_05/lake_sunny_daynight/image/" + img_filename)
            img_array = np.array(img)
            predictions = []
            for transf in combo:
                augmented = apply_augm(transf, img_array)
                input_image = torchvision.transforms.ToTensor()(augmented)
                predictions.append(model(input_image).item())
            
            #variance
            variance_value.append(statistics.variance(predictions))
        
        max_y = max(variance_value)
        chunk_size = 500
        for i in range(0, len(variance_value), chunk_size):
            xpoints = np.arange(i, i+ chunk_size)
            ypoints = np.array(variance_value[i: i+chunk_size])
            x0 = [num for num in index_daynight if (num >= i and num<=(i+chunk_size))]
            y0 = [variance_value[num] for num in x0]
            plt.rcParams["figure.figsize"] = (10,6)
            plt.plot(xpoints, ypoints)
            plt.plot(x0, y0, "o")
            plt.axhline(y=T, color='r', linestyle='-')
            plt.ylim(0, max_y)
            plt.title(" daynight - " + str(combo))
            plt.savefig("plot/ours/new_data/night/" + num_serie +"_plot_" +str(i)+".png")
            plt.clf()
        
        #the first two numbers delimit the night,the second the day..
        segmentation = [189, 389, 489 (513), 689 (663), 789, 989, 1089 (1113), 1289 (1263), 1389, 1589, 1684 (1690), 1884 (1840)]
        j=0 
        print(str(combo))
        for i in range(0, len(segmentation), 2):
            x1points = np.arange((j)*200, (j+1)*200)
            y2points = np.array(variance_value[segmentation[i]: segmentation[i+1]])
            x0 = [num for num in index_daynight if (num >= segmentation[i] and num<segmentation[i+1])]
            y0 = [variance_value[num] for num in x0]
            x0_new = [element - 189 - (j*100) for element in x0]
            plt.rcParams["figure.figsize"] = (10,6)
            plt.plot(x1points, y2points)
            plt.plot(x0_new, y0, "o")
            plt.axhline(y=T, color='r', linestyle='-')
            plt.ylim(0, max_y)
            if j%2==0:
                ttl= "night -"
            else:
                ttl="day -" 
            plt.title(str(combo))
            plt.savefig("plot/ours/new_data/night_v2/" + num_serie +"_plot_" +str(j)+".png")
            plt.clf()
            j= j+1 
        
        k= k+1
            
    