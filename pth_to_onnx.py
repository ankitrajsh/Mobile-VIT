from torch.autograd import Variable
import torch.onnx
import torchvision
import torch 

dummy_input = Variable(torch.randn(1, 3, 256, 256))
model = torch.load('/home/ankit/Downloads/mobilevit-small_3rdparty_in1k_20221018-cb4f741c.pth')
torch.onnx.export(model, dummy_input, "moment-in-time.onnx")`
