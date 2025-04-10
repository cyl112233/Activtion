import torch
import torch.nn as nn
from torchvision.models import  vgg16




class VGG16_model:
    def __init__(self,class_num,activation,pretrained=True,device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")):
        super().__init__()
        self.activation = activation
        self.pretrained = pretrained
        self.class_num = class_num
    def _set_module(self,model, submodule_key, module):
        tokens = submodule_key.split('.')
        sub_tokens = tokens[:-1]
        cur_mod = model
        for s in sub_tokens:
            cur_mod = getattr(cur_mod, s)
        setattr(cur_mod, tokens[-1], module)
    def VGG16(self):
        if self.pretrained:
            model = vgg16(weights=VGG16_Weights.IMAGENET1K_V1)
        else:
            model = vgg16(weights=VGG16_Weights.DEFAULT)

        output_layer = torch.nn.Linear(4096, self.class_num).to(self.device)
        for i in features:
            self._set_module(model, f'features.{i}',self.activation)
        model.classifier[-1] = output_layer
        return model.to(self.device)
from torchvision import models
class DenseNet121(nn.Module):
    #最小尺寸64*64
    def __init__(self, num_classes=1000, activation=EELU2, pretrained=True):
        super(DenseNet121, self).__init__()

 
        if pretrained:
            self.densenet = models.densenet121(weights=models.DenseNet121_Weights.IMAGENET1K_V1)
        else:
            self.densenet = models.densenet121(weights=models.DenseNet121_Weights.DEFAULT)


        self.densenet.classifier = nn.Linear(self.densenet.classifier.in_features, num_classes)

    def _replace_activation(self, module, activation_func):
        """
     
        """
        for child_name, child in module.named_children():
            if isinstance(child, nn.ReLU):
          
                setattr(module, child_name, activation_func())
            else:
               
                self._replace_activation(child, activation_func)

    def forward(self, x):
        return self.densenet(x)




