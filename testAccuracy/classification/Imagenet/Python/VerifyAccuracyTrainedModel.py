# This file is used to test pytorch resnet50 model with pytroch


import torch
from torchvision import datasets, models, transforms
import os
from PIL import Image

def unravel_index(index, shape):
    out = []
    for dim in reversed(shape):
        out.append(index % dim)
        index = index // dim
    return tuple(reversed(out))



PATH = "/home/aistudios/Develop/ai4prod/classes/imagenet/Val/ILSVRC2012_img_val/"

print(torch.cuda.is_available())


trans = transforms.Compose([
            transforms.Resize((256, 256)),
            #transforms.Grayscale(num_output_channels=1),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            #transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])



model= models.resnet50(pretrained=True)


torch.save(model, "resnet.pt")

model.eval()
model.cuda()
images=os.listdir(PATH)

input("test")
import csv
with open('classification-Detection-Python33.csv', 'a', newline='') as file:


    for i,path_image in enumerate(images):

        print(i)
        print(path_image)
        #img= Image.open(PATH + path_image).convert('RGB')

        img=Image.open("/home/aistudios/Develop/ai4prod/test/classification/dog.jpeg")


        tensor= trans(img).float()
        tensor=tensor.unsqueeze(0)
        
        imgNew = transforms.ToPILImage()(tensor.squeeze_(0))
        imgNew.show()
        imgNew.save("dog.jpeg")

        input("test")
        tensor=tensor.cuda()
        


        output= model(tensor)

        top1= torch.argmax(output)
        output=output.view(1000)

        # get best 5 neural network score= return indices
        topk= output.topk(5,0)

        # print("TOpk ",topk)

        
        # print("topk 1 ", topk[0])
        
        # print("topk 2 ", topk[1][0].item())

        

        writer = csv.writer(file)



        image_wihoutExt= os.path.splitext(path_image)[0]

        writer.writerow([image_wihoutExt, topk[1][0].item() , topk[1][1].item(),topk[1][2].item(),topk[1][3].item(),topk[1][4].item()])

        # print(output.size())
        # input("£")
