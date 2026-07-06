import torch 
import numpy
import os
os.environ["TORCH_CUDA_ARCH_LIST"] = "5.2"
# Continue with Sebastian's book code


tensor1d=torch.tensor([1.0, 2.0, 3.0, 4, 5])
print(tensor1d.dtype)

floatvec=torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
print(floatvec.dtype)

print(torch.__version__)
print(numpy.__version__)

print(torch.cuda.is_available())

#Now the main code