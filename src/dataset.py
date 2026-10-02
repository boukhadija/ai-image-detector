# Dataset loading and preprocessing
from PIL import Image
from torchvision import transforms
image_path =("data/train/real/test_image.jpg")
image_test=Image.open(image_path)
print(image_test.size)
print(image_test.mode)
resized_image=image_test.resize((224,224))
print(resized_image.size)
to_tensor=transforms.ToTensor()
tensor_image=to_tensor(resized_image)
print(tensor_image.shape)