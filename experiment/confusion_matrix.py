import statistics
import pandas
import itertools


def metrics(set_transformation, n, treshold):
    
    # Generate all combinations of n elements
    combinations = list(itertools.combinations(set_transformation, n))
    
    cost = []
    # night over treshold
    true_positive = []
    dist_tp = []
    # day under treshold
    true_negative = []
    dist_tn = []
    # day over treshold
    false_positive = []
    dist_fp = []
    # night under treshold
    false_negative = []
    dist_fn = []
    
    precision = []
    recall = []
    f1_score = []
    
    for combo in combinations:
        print(combo)
        #normal one
        #df = pandas.read_csv( "~/work/code/udacity-gym/variance_augm/new_data/" + str(n) + "_transf/day-" + str(combo) + ".csv")
        
        #window avarage
        df = pandas.read_csv( "~/work/code/udacity-gym/variance_augm/new_data/" + str(n) + "_transf/variance_mw/" + str(combo) + ".csv")
        
        #variance_day = df['Variance']
        variance_day = df['Variance Day']
        T = variance_day.sort_values().iloc[int(len(variance_day) * treshold)]
        print(T)
        
        #df = pandas.read_csv( "~/work/code/udacity-gym/variance_augm/new_data/" + str(n) + "_transf/daynight-" + str(combo) + ".csv")
        #variance_daynight = df['Variance']
        
        #window avarage
        variance_daynight = df['Variance Day Night']
        
        tp=0
        tn=0
        fn=0
        fp=0
        dtp=0
        dtn=0
        dfn=0
        dfp=0
        
        #strart from night
        #segmentation = [189, 389, (513), (663), 789, 989, (1113), (1263), 1389, 1589, (1690), (1840)]
        for i in range(200):
            if variance_daynight.iloc[188+i] >= T:
                tp = tp + 1
                dtp = dtp + (variance_daynight.iloc[188+i] - T)
            else: 
                fn = fn + 1
                dfn = dfn + (T - variance_daynight.iloc[188+i])
        
        for i in range(150):
            if variance_daynight.iloc[512+i] < T:
                tn = tn + 1
                dtn = dtn + (T - variance_daynight.iloc[512+i])
            else: 
                fp = fp + 1
                dfp = dfp + (variance_daynight.iloc[512+i] - T)
        
        for i in range(200):
            if variance_daynight.iloc[788+i] >= T:
                tp = tp + 1
                dtp = dtp + (variance_daynight.iloc[788+i] - T)
            else: 
                fn = fn + 1
                dfn = dfn + (T - variance_daynight.iloc[788+i])
        
        for i in range(150):
            if variance_daynight.iloc[1112+i] < T:
                tn = tn + 1
                dtn = dtn + (T - variance_daynight.iloc[1112+i])
            else: 
                fp = fp + 1
                dfp = dfp + (variance_daynight.iloc[1112+i] - T)
        
        for i in range(200):
            if variance_daynight.iloc[1388+i] >= T:
                tp = tp + 1
                dtp = dtp + (variance_daynight.iloc[1388+i] - T)
            else: 
                fn = fn + 1
                dfn = dfn + (T - variance_daynight.iloc[1388+i])
        
        for i in range(150):
            if variance_daynight.iloc[1689+i] < T:
                tn = tn + 1
                dtn = dtn + (T - variance_daynight.iloc[1689+i])
            else: 
                fp = fp + 1
                dfp = dfp + (variance_daynight.iloc[1689+i] - T)
        
        if (tp + fp)!=0:
            pr = tp / (tp + fp)
        else:
            pr=0
        
        if (tp + fn)!=0:
            rc = tp / (tp + fn)
        else: 
            rc=0
        
        if tp!=0:
            f1 = 2 * (pr*rc)/(pr+rc)
        else:
            f1 = 0
        
        true_positive.append(tp)
        true_negative.append(tn)
        false_positive.append(fp)
        false_negative.append(fn)
        precision.append(pr)
        recall.append(rc)
        f1_score.append(f1)
        cost.append(dtp + dtn - dfp - dfn)
        
        
    
    # Step 4: Create a DataFrame from the results
    df = pandas.DataFrame({
        'Name': combinations,
        'True Postive': true_positive,
        'True Negative': true_negative,
        'False Positive': false_positive,
        'False Negative': false_negative,
        'Precision': precision,
        'Recall': recall,
        'F1 Score': f1_score,
        'Cost': cost
    })
    
    # Step 5: Save the DataFrame to a CSV file
    #df.to_csv('metrics/new_data/'+ str(n)+'_transf_'+ str(int(treshold*100)) + '.csv', index=False)
    df.to_csv('metrics/new_data/mw_'+ str(n)+'_transf_'+ str(int(treshold*100)) + '.csv', index=False)

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
    
    #metrics(set_transformation, 3, 0.95)
    #metrics(set_transformation, 3, 0.90)
    #metrics(set_transformation, 3, 0.75)
    #metrics(set_transformation, 4, 0.95)
    #metrics(set_transformation, 4, 0.90)
    #metrics(set_transformation, 4, 0.75)
    metrics(set_transformation, 5, 0.95)
    metrics(set_transformation, 5, 0.90)
    metrics(set_transformation, 5, 0.75)
    print("Experiment concluded")