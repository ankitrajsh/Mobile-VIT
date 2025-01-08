import requests
import torch
from PIL import Image
from transformers import AutoImageProcessor, MobileViTForSemanticSegmentation

url = "http://images.cocodataset.org/val2017/000000039769.jpg"
image = Image.open(requests.get(url, stream=True).raw)

image_processor = AutoImageProcessor.from_pretrained("apple/deeplabv3-mobilevit-small")
model = MobileViTForSemanticSegmentation.from_pretrained("apple/deeplabv3-mobilevit-small")

inputs = image_processor(images=image, return_tensors="pt")

with torch.no_grad():
    outputs = model(**inputs)

# logits are of shape (batch_size, num_labels, height, width)
logits = outputs.logits