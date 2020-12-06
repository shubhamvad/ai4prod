import os
import xml.etree.ElementTree as ET

import numpy as np

import pandas as pd 

# questo file genera un csv con id_immagine e classe per il validation

# converte il numero delle classi con output rete neurale


DATASET_LABELS= "/home/aistudios/Develop/ai4prod/classes/imagenet/Val/ILSVRC2012_bbox_val_v3/val/"

labels= os.listdir(DATASET_LABELS)


#load json classes

df= pd.read_json("imagenet_class_index.json")


print(df.head())

arrayLabels=[]


f = open("validation-imagenet.csv", "w+")


for label in labels:
    
    array_single_label=[]

    print(label)
    tree = ET.parse(DATASET_LABELS+ label)
    root = tree.getroot()
    
    print(root[5][0].text)
    print(root[1].text)

    result=df.isin([root[5][0].text])
    seriesObj = result.any()

    
    columnNames=list(seriesObj[seriesObj == True].index)


    
    str_to_write= root[1].text +"," + str(columnNames[0]) + " \n"
    
    f.write(str_to_write)
    
    print('Names of columns which contains 81:', columnNames[0])




f.close()


