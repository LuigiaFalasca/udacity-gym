import json
import pandas
import PIL
import numpy as np
import pandas
from model.lane_keeping.dave.dave_model_v2 import Dave2
import statistics
import matplotlib.pyplot as plt
import pathlib
import torchvision


if __name__ == '__main__':
    with open("new_data/udacity_dataset_lake_dave_05/lake_sunny_day/info.json") as f:
        data = json.load(f)
    '''
    collisions = []
    for i in range(1, len(data['events'])):
        collisions.append(data['events'][i]['timestamp'])
    
    #timestamp before the collision
    index= []
    timestamps=[]
    
    df = pandas.read_csv("~/work/code/udacity-gym/new_data/udacity_dataset_lake_dave_05/lake_sunny_day/log.csv")
    j=0
    for i in range(len(df['time'])-1):
        if j < len(collisions):
            if (int(collisions[j])>= int(df['time'][i]) and int(collisions[j])<= int(df['time'][i+1])):
                timestamps.append(
                    {'collision_tmp': collisions[j],
                     'before_tmp': df['time'][i],
                     'after_tmp': df['time'][i+1],
                     'before_index': i,
                     'after_index': i+1
                    }
                )
                j= j+1
                index.append(i)
    
    '''
    #get the nn
    checkpoint_path = pathlib.Path("~/work/code/udacity-gym/new_data/dave2-v2.ckpt")
    model = Dave2.load_from_checkpoint(checkpoint_path)
    model.eval()
    model = model.to("cuda:0")
    
    #get the dropout setting
    for m in model.modules():
        if m.__class__.__name__.startswith('Dropout'):
            m.train()
    
    
    variance_value=[]
    #get images from the folder
    df = pandas.read_csv("~/work/code/udacity-gym/new_data/udacity_dataset_lake_dave_05/lake_sunny_day/log.csv")
    
    for img_filename in df['image_filename']:
        img = PIL.Image.open("new_data/udacity_dataset_lake_dave_05/lake_sunny_day/image/" + img_filename)
        input_image = torchvision.transforms.ToTensor()(img)
        input_image = input_image.to("cuda:0")
        #dropout prediction
        N = 32
        # make the predictions
        predictions = []
        for _ in range(N):
            predictions.append(model(input_image).item())
        #variance
        variance_value.append(statistics.variance(predictions))
    
    T = sorted(variance_value)[int(len(variance_value) * 0.95)]
    '''
    max_y = max(variance_value)
    chunk_size = 500
    for i in range(0, len(variance_value), chunk_size):
        xpoints = np.arange(i, i+ chunk_size)
        ypoints = np.array(variance_value[i: i+chunk_size])
        x0 = [num for num in index if (num >= i and num<=(i+chunk_size))]
        y0 = [variance_value[num] for num in x0]
        plt.rcParams["figure.figsize"] = (10,6)
        plt.plot(xpoints, ypoints)
        plt.plot(x0, y0, "o")
        plt.axhline(y=T, color='r', linestyle='-')
        plt.ylim(0, max_y)
        plt.title("Dropout - day ")
        plt.savefig("plot/dropout/new_data/day/plot" +str(i)+".png")
        plt.clf()
    
    '''
    #start with daynight
    with open("new_data/udacity_dataset_lake_dave_05/lake_sunny_daynight/info.json") as f:
        data = json.load(f)
    '''
    collisions = []
    for i in range(1, len(data['events'])):
        collisions.append(data['events'][i]['timestamp'])
    
    #timestamp before the collision
    index= []
    timestamps=[]
    
    df = pandas.read_csv("~/work/code/udacity-gym/new_data/udacity_dataset_lake_dave_05/lake_sunny_daynight/log.csv")
    j=0
    for i in range(len(df['time'])-1):
        if j < len(collisions):
            if (int(collisions[j])>= int(df['time'][i]) and int(collisions[j])<= int(df['time'][i+1])):
                timestamps.append(
                    {'collision_tmp': collisions[j],
                     'before_tmp': df['time'][i],
                     'after_tmp': df['time'][i+1],
                     'before_index': i,
                     'after_index': i+1
                    }
                )
                j= j+1
                index.append(i)
    '''
    variance_daynight=[]
    #get images from the folder
    df = pandas.read_csv("~/work/code/udacity-gym/new_data/udacity_dataset_lake_dave_05/lake_sunny_daynight/log.csv")
    
    for img_filename in df['image_filename']:
        img = PIL.Image.open("new_data/udacity_dataset_lake_dave_05/lake_sunny_daynight/image/" + img_filename)
        input_image = torchvision.transforms.ToTensor()(img)
        input_image = input_image.to("cuda:0")
        #dropout prediction
        N = 32
        # make the predictions
        predictions = []
        for _ in range(N):
            predictions.append(model(input_image).item())
        #variance
        variance_daynight.append(statistics.variance(predictions))
    '''
    max_y = max(variance_value)
    chunk_size = 500
    for i in range(0, len(variance_value), chunk_size):
        xpoints = np.arange(i, i+ chunk_size)
        ypoints = np.array(variance_value[i: i+chunk_size])
        x0 = [num for num in index if (num >= i and num<=(i+chunk_size))]
        y0 = [variance_value[num] for num in x0]
        plt.rcParams["figure.figsize"] = (10,6)
        plt.plot(xpoints, ypoints)
        plt.plot(x0, y0, "o")
        plt.axhline(y=T, color='r', linestyle='-')
        plt.ylim(0, max_y)
        plt.title("Dropout - daynight")
        plt.savefig("plot/dropout/new_data/night/plot" +str(i)+".png")
        plt.clf()
    
    #the first two numbers delimit the night,the second the day..
    segmentation = [189, 389, 489, 689, 789, 989, 1089, 1289, 1389, 1589, 1684, 1884]
    j=0 
    for i in range(0, len(segmentation), 2):
        x1points = np.arange((j)*200, (j+1)*200)
        y2points = np.array(variance_value[segmentation[i]: segmentation[i+1]])
        x0 = [num for num in index if (num >= segmentation[i] and num<segmentation[i+1])]
        y0 = [variance_value[num] for num in x0]
        x0_new = [element - 189 - (j*100) for element in x0]
        plt.rcParams["figure.figsize"] = (10,6)
        plt.plot(x1points, y2points)
        plt.plot(x0_new, y0, "o")
        plt.axhline(y=T, color='r', linestyle='-')
        plt.ylim(0, max_y)
        if j%2==0:
            ttl= "night"
        else:
            ttl="day"
        plt.title("Dropout - "+ ttl)
        plt.savefig("plot/dropout/new_data/night_v2/plot_" +str(j)+".png")
        plt.clf()
        j= j+1 
    
    '''
    tp=0
    tn=0
    fn=0
    fp=0
    dtp=0
    dtn=0
    dfn=0
    dfp=0
    
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
    #strart from night
    #segmentation = [189, 389, (513), (663), 789, 989, (1113), (1263), 1389, 1589, (1690), (1840)]
    '''
    for i in range(200):
        if variance_daynight[188+i] >= T:
            tp = tp + 1
            dtp = dtp + (variance_daynight[188+i] - T)
        else: 
            fn = fn + 1
            dfn = dfn + (T - variance_daynight[188+i])
    '''   
    for i in range(150):
        if variance_daynight[512+i] < T:
            tn = tn + 1
            dtn = dtn + (T - variance_daynight[512+i])
        else: 
            fp = fp + 1
            dfp = dfp + (variance_daynight[512+i] - T)
        
    for i in range(200):
        if variance_daynight[788+i] >= T:
            tp = tp + 1
            dtp = dtp + (variance_daynight[788+i] - T)
        else: 
            fn = fn + 1
            dfn = dfn + (T - variance_daynight[788+i])
        
    for i in range(150):
        if variance_daynight[1112+i] < T:
            tn = tn + 1
            dtn = dtn + (T - variance_daynight[1112+i])
        else: 
            fp = fp + 1
            dfp = dfp + (variance_daynight[1112+i] - T)
        
    for i in range(200):
        if variance_daynight[1388+i] >= T:
            tp = tp + 1
            dtp = dtp + (variance_daynight[1388+i] - T)
        else: 
            fn = fn + 1
            dfn = dfn + (T - variance_daynight[1388+i])
        
    for i in range(150):
        if variance_daynight[1689+i] < T:
            tn = tn + 1
            dtn = dtn + (T - variance_daynight[1689+i])
        else: 
            fp = fp + 1
            dfp = dfp + (variance_daynight[1689+i] - T)
        
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
    df.to_csv('metrics/new_data/dropout_95_without_fs.csv', index=False)
    
    T = sorted(variance_value)[int(len(variance_value) * 0.90)]
    
    tp=0
    tn=0
    fn=0
    fp=0
    dtp=0
    dtn=0
    dfn=0
    dfp=0
    
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
    #strart from night
    #segmentation = [189, 389, (513), (663), 789, 989, (1113), (1263), 1389, 1589, (1690), (1840)]
    '''
    for i in range(200):
        if variance_daynight[188+i] >= T:
            tp = tp + 1
            dtp = dtp + (variance_daynight[188+i] - T)
        else: 
            fn = fn + 1
            dfn = dfn + (T - variance_daynight[188+i])
    '''   
    for i in range(150):
        if variance_daynight[512+i] < T:
            tn = tn + 1
            dtn = dtn + (T - variance_daynight[512+i])
        else: 
            fp = fp + 1
            dfp = dfp + (variance_daynight[512+i] - T)
        
    for i in range(200):
        if variance_daynight[788+i] >= T:
            tp = tp + 1
            dtp = dtp + (variance_daynight[788+i] - T)
        else: 
            fn = fn + 1
            dfn = dfn + (T - variance_daynight[788+i])
        
    for i in range(150):
        if variance_daynight[1112+i] < T:
            tn = tn + 1
            dtn = dtn + (T - variance_daynight[1112+i])
        else: 
            fp = fp + 1
            dfp = dfp + (variance_daynight[1112+i] - T)
        
    for i in range(200):
        if variance_daynight[1388+i] >= T:
            tp = tp + 1
            dtp = dtp + (variance_daynight[1388+i] - T)
        else: 
            fn = fn + 1
            dfn = dfn + (T - variance_daynight[1388+i])
        
    for i in range(150):
        if variance_daynight[1689+i] < T:
            tn = tn + 1
            dtn = dtn + (T - variance_daynight[1689+i])
        else: 
            fp = fp + 1
            dfp = dfp + (variance_daynight[1689+i] - T)
        
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
    df.to_csv('metrics/new_data/dropout_90_without_fs.csv', index=False)
    
    
    T = sorted(variance_value)[int(len(variance_value) * 0.75)]
    
    tp=0
    tn=0
    fn=0
    fp=0
    dtp=0
    dtn=0
    dfn=0
    dfp=0
    
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
    #strart from night
    #segmentation = [189, 389, (513), (663), 789, 989, (1113), (1263), 1389, 1589, (1690), (1840)]
    '''
    for i in range(200):
        if variance_daynight[188+i] >= T:
            tp = tp + 1
            dtp = dtp + (variance_daynight[188+i] - T)
        else: 
            fn = fn + 1
            dfn = dfn + (T - variance_daynight[188+i])
    '''   
    for i in range(150):
        if variance_daynight[512+i] < T:
            tn = tn + 1
            dtn = dtn + (T - variance_daynight[512+i])
        else: 
            fp = fp + 1
            dfp = dfp + (variance_daynight[512+i] - T)
        
    for i in range(200):
        if variance_daynight[788+i] >= T:
            tp = tp + 1
            dtp = dtp + (variance_daynight[788+i] - T)
        else: 
            fn = fn + 1
            dfn = dfn + (T - variance_daynight[788+i])
        
    for i in range(150):
        if variance_daynight[1112+i] < T:
            tn = tn + 1
            dtn = dtn + (T - variance_daynight[1112+i])
        else: 
            fp = fp + 1
            dfp = dfp + (variance_daynight[1112+i] - T)
        
    for i in range(200):
        if variance_daynight[1388+i] >= T:
            tp = tp + 1
            dtp = dtp + (variance_daynight[1388+i] - T)
        else: 
            fn = fn + 1
            dfn = dfn + (T - variance_daynight[1388+i])
        
    for i in range(150):
        if variance_daynight[1689+i] < T:
            tn = tn + 1
            dtn = dtn + (T - variance_daynight[1689+i])
        else: 
            fp = fp + 1
            dfp = dfp + (variance_daynight[1689+i] - T)
        
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
    df.to_csv('metrics/new_data/dropout_75_without_fs.csv', index=False)