import numpy as np
import pandas
import itertools


def window(set_transformation, n, w):

    # Generate all combinations of 5 elements
    combinations = list(itertools.combinations(set_transformation, n))
    
    for combo in combinations:
        print(combo)
        df = pandas.read_csv( "~/work/code/udacity-gym/variance_augm/new_data/"+ str(n) + "_transf/day-" + str(combo) + ".csv")
        variance_day = df['Variance']
        variance_day.rolling(w, min_periods=1).mean()

        df = pandas.read_csv( "~/work/code/udacity-gym/variance_augm/new_data/"+ str(n) + "_transf/daynight-" + str(combo) + ".csv")
        variance_daynight = df['Variance']
        variance_daynight.rolling(w, min_periods=1).mean()
        
        df = pandas.DataFrame({
            'Variance Day': variance_day,
            'Variance Day Night': variance_daynight,
        })
        
        df.to_csv('variance_augm/new_data/' + str(n) + '_transf/variance_mw/'+ str(w)+ '_' + str(combo) +'.csv', index=False)

if __name__ == '__main__':
    
    set_transformation = [
        "ToGray",  "ChannelShuffle", "RGBShift", "Blur",
        "HueSaturationValue", "ChannelDropout", "Downscale", "CLAHE",
        "Defocus",  "ColorJitter", "Equalize", "InvertImg",
        "PlanckianJitter", "ZoomBlur", "Sharpen", "RingingOvershoot",
        "Superpixels", "Spatter", "ToSepia", "Solarize",
        "Posterize", "RandomBrightnessContrast", "RandomFog",
        "RandomShadow", "RandomGamma", "RandomSnow", "RandomRain"
    ]
    
    #window(set_transformation, 3)
    #window(set_transformation, 4)
    window(set_transformation, 5, 2)
    window(set_transformation, 5, 4)
    window(set_transformation, 5, 5)
    window(set_transformation, 5, 10)
    
    print("Experiment concluded")