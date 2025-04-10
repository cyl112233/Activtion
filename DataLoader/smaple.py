from dataloader import get_dataloader
import torch
import random
import numpy as np
Train,Test,LenData,num_class = get_dataloader('CIFAR10',1)
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("tkagg")
index = 0
task_list = []
while True:
    num = random.randint(0,num_class-1)
    if num not in task_list:
        task_list.append(num)
    if len(task_list) == 10:
        break
class_names = [
    'airplane', 'automobile', 'bird', 'cat', 'deer',
    'dog', 'frog', 'horse', 'ship', 'truck'
]
#
for image,label in Train:
    if label.item() == task_list[index]:
        plt.subplot(2,5,index+1)
        plt.axis('off')
        image = np.swapaxes(np.array(torch.squeeze(image,0)),0,2).swapaxes(0,1)
        plt.imshow(image)
        plt.title(class_names[label])
        plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1, hspace=0.2)
        index += 1
   

plt.show()
