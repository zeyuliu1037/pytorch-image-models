import torch

model = torch.load('output/train/20221026-052153-twins_pcpvt_small_spike-224/model_best.pth.tar'
                   , map_location=torch.device('cpu'))
# print(model.keys()) dict_keys(['epoch', 'arch', 'state_dict', 'optimizer', 'version', 'args', 'amp_scaler', 'metric'])

for k, v in model['state_dict'].items():
    print(k, v.shape, v.min(), v.max())