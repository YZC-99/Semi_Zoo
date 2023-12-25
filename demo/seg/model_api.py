import segmentation_models_pytorch as smp
import torch
# model = smp.Unet(
#     encoder_name = 'resnet34',
#     encoder_weights = 'imagenet',
#     in_channels = 3,
#     classes=3
# )
model = smp.DeepLabV3Plus(
    encoder_name = 'resnet34',
    encoder_weights = 'imagenet',
    in_channels = 3,
    classes=3
)
data = torch.randn(4,3,512,512)
out = model(data)
print(out.shape)