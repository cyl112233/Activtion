

from torchvision.models import densenet121
import torch
# from torchsummary import summary
print(densenet121(pretrained=True))
# def _set_module(model, submodule_key, module):
#     tokens = submodule_key.split('.')
#     sub_tokens = tokens[:-1]
#     cur_mod = model
#     for s in sub_tokens:
#         cur_mod = getattr(cur_mod, s)
#     setattr(cur_mod, tokens[-1], module)
#
#
# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# model = vgg16(pretrained=True).to(device)
# Ativation_layer = torch.nn.LeakyReLU()
# output_layer = torch.nn.Linear(4096, 10)
#
# features = [1,3,6,8,11,13,15,18,20,22,25,27,29]
# classifier = [1,4,6]
#
# for module in model.named_modules():
#     print(module)
# for i in features:
#     _set_module(model, f'features.{i}', Ativation_layer)
# for i in classifier:
#     _set_module(model, f'classifier.{i}', Ativation_layer)
#     if i == 6:
#         _set_module(model, f'classifier.{i}', output_layer)


#
# _set_module(model, 'features.3', Actvtion_layer)
# _set_module(model, 'features.5', Actvtion_layer)
# torch.save(model,"VGG16.pt")
