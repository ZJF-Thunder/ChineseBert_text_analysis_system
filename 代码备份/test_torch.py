import torch
# CUDA_VISIBLE_DEVICES = 1
print("cuda是否可用:", torch.cuda.is_available())   # cuda是否可用
print("cuda版本:", torch.version.cuda)  # cuda版本
print("当前设备索引:", torch.cuda.current_device())   # 返回当前设备索引
print("GPU数量:", torch.cuda.device_count())    # 返回GPU的数量
print("GPU名字:", torch.cuda.get_device_name(0))   # 返回gpu名字，设备索引默认从0开始

print("torch版本:", torch.__version__)

print("torch的cuda版本:", torch.version.cuda)

print("cudnn版本:", torch.backends.cudnn.version())