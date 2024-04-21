import torch

from module.models import SynthesizerTrn
from vc_main import VC

net = VC(
    inter_channels=192,
    hidden_channels=192,
    filter_channels=768,
    n_heads=2,
    n_layers=6,
    kernel_size=3,
    p_dropout=0,
    spec_channels=1025,
    resblock='1',
    resblock_kernel_sizes=[3, 7, 11],
    resblock_dilation_sizes=[[1, 3, 5], [1, 3, 5], [1, 3, 5]],
    upsample_rates=[10, 8, 2, 2, 2],
    upsample_initial_channel=512,
    upsample_kernel_sizes=[16, 16, 8, 2, 2],
    gin_channels=512
)
# 修改前
dict = torch.load('pretvc_model.pth')
net.load_state_dict(dict)
# for name, param in net.named_parameters():
#     print(name)

dict_s2 = torch.load("./GPT_weights/Nahida_e8_s448.pth", map_location="cpu")
hps = dict_s2["config"]
from test import DictToAttrRecursive

hps = DictToAttrRecursive(hps)
hps.model.semantic_frame_rate = "25hz"
vq_model = SynthesizerTrn(
    hps.data.filter_length // 2 + 1,
    hps.train.segment_size // hps.data.hop_length,
    n_speakers=hps.data.n_speakers,
    **hps.model
)
vq_model.load_state_dict(dict_s2["weight"], strict=False)
vq_layer = []
for name, param in vq_model.named_parameters():
    # print(name)
    vq_layer.append(name)

# 修改权重
for name, param in net.named_parameters():
    if name not in dict_s2['weight']:
        print("============")
        print(name)
    else:
        dict[name] = dict_s2['weight'][name]
        # print("++++++++++",name,"+++++++++++")

# #按参数名修改权重
# dict["forward1.0.weight"] = torch.ones((1,1,3,3,3))
# dict["forward1.0.bias"] = torch.ones(1)
torch.save(dict, 'vc_model.pth')
# #验证修改是否成功
# net.load_state_dict(torch.load('./ckpt_dir//model_0_.pth'))
# for param_tensor in net.state_dict():
# 	print(net.state_dict()[param_tensor])
