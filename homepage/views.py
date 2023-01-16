from django.http import HttpResponse
from django.shortcuts import render
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader  # For working with data.
# For pretrained models,image transformations.
from django.core.files.storage import FileSystemStorage
from torchvision import models as modelsNN, transforms
import queue
import torchvision.transforms.functional as TF
from torch.utils.data import IterableDataset
from PIL import Image
from . forms import *
import cv2

# Create your views here.
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

loaded_model = modelsNN.resnet34(pretrained=True) #Downloads the resnet18 model which is pretrained on Imagenet dataset.

#Replace the Final layer of pretrained resnet18 with 4 new layers.
loaded_model.fc = nn.Sequential(nn.Linear(512,256),nn.Linear(256,128),nn.Linear(128,64),nn.Linear(64,5))
#for webapp
loaded_model = loaded_model.to(device) #Moves the model to the device.
#for webapp
loaded_model.load_state_dict(torch.load('Models/model.pth', map_location=torch.device('cpu')))
#for weba



def index(request):
    res=""
    # form = ImageForm()
    # ans = -1
    # if request.method=='POST':
    #         form = ImageForm(request.POST)
    #         if form.is_valid():
    #             form.save()
    #             # ip_image = obj.img
    #             # ans = getOutput(ip_image)
    #             # return redirect("home")
    # context = {'form':form}
    if (len(request.FILES) != 0):
      
        fileObj=request.FILES['filePath']
        print(fileObj)
        fs=FileSystemStorage()
        filePathName=fs.save(fileObj.name,fileObj)
        filePathName=fs.url(filePathName)
        print(filePathName)
        testimage='static/'+filePathName
        imageDisplay = 'static/'+filePathName[1:]
        # print(testimage)
        class MyDataset(IterableDataset):
            def __init__(self, image_queue):
                self.queue = image_queue

            def read_next_image(self):
                while self.queue.qsize() > 0:
                    # you can add transform here 
                    yield self.queue.get()
                return None

            def __iter__(self):
                return self.read_next_image()

       
        image_transform = transforms.Compose([transforms.Resize([512,512]),
                                            transforms.ToTensor(),
                                            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))]) #Transformations to apply to the image.
        
       
        buffer = queue.Queue()
        # new_input = cv2.imread(testimage)
        new_input = Image.open(testimage)
        # buffer.put(TF.to_tensor(new_input)) 
        # # # # ... Populate queue here
        
        buffer.put(image_transform(new_input))
        dataset = MyDataset(buffer)

        new_dataloader = torch.utils.data.DataLoader(dataset, batch_size=1)
        loaded_model.eval()

        with torch.no_grad():
            for batch,x in enumerate(new_dataloader):
                output = loaded_model(x.to(device))
                
                predictions = output.argmax(dim=1).cpu().detach().tolist() #Predicted labels for an image batch. 
                
            print('Testing has completed')
            print(predictions[0])
            ans = predictions[0]
            
            if ans == 0:
                res = "Normal"
            elif ans == 1:
                res = "Mild"
            elif ans == 2:
                res ="Moderate"
            elif ans == 3:
                res = "Severe"
            else:
                res = "Proliferative"
        context = {'ans':res,'imgPath':imageDisplay}
        # dispResult(request,res)
        return render(request,"homepage/result.html",context)
    return render(request,"homepage/index.html")

# def dispResult(request, result):
#     context = {'ans':result}
#     return render(request,"homepage/result.html",context)


