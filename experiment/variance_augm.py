import statistics
import pandas
import itertools


def varAugm(set_transformation, n):

    # Generate all combinations of n elements
    combinations = list(itertools.combinations(set_transformation, n))
    
    for combo in combinations:
        print(combo)
        columns = []
        for transf in combo:
            path = "~/work/code/udacity-gym/prediction_augm/new_data/" + transf + ".csv"
            df = pandas.read_csv(path)
            columns.append(df['Day'])
        
        combined_df = pandas.concat(columns, axis=1)
        variance_series = combined_df.var(axis=1)
        combined_df['Variance'] = variance_series
        combined_df.to_csv('variance_augm/new_data/'+ str(n) + '_transf/day-'+ str(combo) +'.csv', index=False)
        
        columns = []
        for transf in combo:
            path = "~/work/code/udacity-gym/prediction_augm/new_data/" + transf + ".csv"
            df = pandas.read_csv(path)
            columns.append(df['DayNight'])
        
        combined_df = pandas.concat(columns, axis=1)
        variance_series = combined_df.var(axis=1)
        combined_df['Variance'] = variance_series
        combined_df.to_csv('variance_augm/new_data/'+ str(n) + '_transf/daynight-'+ str(combo) +'.csv', index=False)
        

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
    
    varAugm(set_transformation, 3)
    varAugm(set_transformation, 4)
    varAugm(set_transformation, 5)
    
    print("Experiment concluded")
