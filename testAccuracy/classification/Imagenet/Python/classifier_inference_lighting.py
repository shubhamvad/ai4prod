import pytorch_lightning as pl
from torchvision import transforms,models
import os
#this file create a csv file ready to be processed with imagnet label in order to get accuracy
#the csv file 

from PIL import Image
import torch

import csv

from models_lighting import *



PATH = "/home/aistudios/Develop/Official/ai4prod/Dataset/Imagenet2012/val2012/"

model_class= ImagenetInference()




model= model_class.model

# torch.save(model.state_dict(), 'resnet50.pth')
input("model")
model.cuda()
model.eval()

images=os.listdir(PATH)

with open('classification-Detection-Python-Lighting.csv', 'a', newline='') as file:
    for i,path_image in enumerate(images):

        img= Image.open(PATH+path_image).convert('RGB')
        
        tensor= model_class.transform(img).float()
        
        tensor= tensor.unsqueeze(0)
        
        output= model(tensor.cuda())
        
        top1= torch.argmax(output)
        
        output=output.view(1000)
        # get best 5 neural network score= return indices
        topk= output.topk(5,0)


        
        print(top1)
       
        
        writer = csv.writer(file)



        image_wihoutExt= os.path.splitext(path_image)[0]

        writer.writerow([image_wihoutExt, topk[1][0].item() , topk[1][1].item(),topk[1][2].item(),topk[1][3].item(),topk[1][4].item()])

        


        