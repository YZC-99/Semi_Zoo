from collections import OrderedDict
from model.backbone.efficient.efficient_layers import  *
from model.backbone.efficient.efficientnet import EfficientNet
from model.module.gap import GlobalAvgPool2D
from model.module import fpn
import torch.nn as nn
from model.module.farseg import SceneRelation

__all__ = ['EfficientUnet', 'get_efficientunet_b0', 'get_efficientunet_b1', 'get_efficientunet_b2',
           'get_efficientunet_b3', 'get_efficientunet_b4', 'get_efficientunet_b5', 'get_efficientunet_b6',
           'get_efficientunet_b7']


def get_blocks_to_be_concat(model, x):
    shapes = set()
    blocks = OrderedDict()
    hooks = []
    count = 0

    def register_hook(module):

        def hook(module, input, output):
            try:
                nonlocal count
                if module.name == f'blocks_{count}_output_batch_norm':
                    count += 1
                    shape = output.size()[-2:]
                    if shape not in shapes:
                        shapes.add(shape)
                        blocks[module.name] = output

                elif module.name == 'head_swish':
                    # when module.name == 'head_swish', it means the program has already got all necessary blocks for
                    # concatenation. In my dynamic unet implementation, I first upscale the output of the backbone,
                    # (in this case it's the output of 'head_swish') concatenate it with a block which has the same
                    # Height & Width (image size). Therefore, after upscaling, the output of 'head_swish' has bigger
                    # image size. The last block has the same image size as 'head_swish' before upscaling. So we don't
                    # really need the last block for concatenation. That's why I wrote `blocks.popitem()`.
                    blocks.popitem()
                    blocks[module.name] = output

            except AttributeError:
                pass

        if (
                not isinstance(module, nn.Sequential)
                and not isinstance(module, nn.ModuleList)
                and not (module == model)
        ):
            hooks.append(module.register_forward_hook(hook))

    # register hook
    model.apply(register_hook)

    # make a forward pass to trigger the hooks
    model(x)

    # remove these hooks
    for h in hooks:
        h.remove()

    return blocks


class EfficientUnet(nn.Module):
    def __init__(self, encoder, out_channels=2, concat_input=True):
        super().__init__()

        self.encoder = encoder
        self.concat_input = concat_input

        self.up_conv1 = up_conv(self.n_channels, 512)
        self.double_conv1 = double_conv(self.size[0], 512)
        self.up_conv2 = up_conv(512, 256)
        self.double_conv2 = double_conv(self.size[1], 256)
        self.up_conv3 = up_conv(256, 128)
        self.double_conv3 = double_conv(self.size[2], 128)
        self.up_conv4 = up_conv(128, 64)
        self.double_conv4 = double_conv(self.size[3], 64)

        if self.concat_input:
            self.up_conv_input = up_conv(64, 32)
            self.double_conv_input = double_conv(self.size[4], 32)

        self.final_conv = nn.Conv2d(self.size[5], out_channels, kernel_size=1)

    @property
    def n_channels(self):
        n_channels_dict = {'efficientnet-b0': 1280, 'efficientnet-b1': 1280, 'efficientnet-b2': 1408,
                           'efficientnet-b3': 1536, 'efficientnet-b4': 1792, 'efficientnet-b5': 2048,
                           'efficientnet-b6': 2304, 'efficientnet-b7': 2560}
        return n_channels_dict[self.encoder.name]

    @property
    def size(self):
        size_dict = {'efficientnet-b0': [592, 296, 152, 80, 35, 32], 'efficientnet-b1': [592, 296, 152, 80, 35, 32],
                     'efficientnet-b2': [600, 304, 152, 80, 35, 32], 'efficientnet-b3': [608, 304, 160, 88, 35, 32],
                     'efficientnet-b4': [624, 312, 160, 88, 35, 32], 'efficientnet-b5': [640, 320, 168, 88, 35, 32],
                     'efficientnet-b6': [656, 328, 168, 96, 35, 32], 'efficientnet-b7': [672, 336, 176, 96, 35, 32]}
        return size_dict[self.encoder.name]

    def forward(self, x):
        input_ = x

        blocks = get_blocks_to_be_concat(self.encoder, x)
        _, x = blocks.popitem()

        x = self.up_conv1(x)
        x = torch.cat([x, blocks.popitem()[1]], dim=1)
        x = self.double_conv1(x)

        x = self.up_conv2(x)
        x = torch.cat([x, blocks.popitem()[1]], dim=1)
        x = self.double_conv2(x)

        x = self.up_conv3(x)
        x = torch.cat([x, blocks.popitem()[1]], dim=1)
        x = self.double_conv3(x)

        x = self.up_conv4(x)
        x = torch.cat([x, blocks.popitem()[1]], dim=1)
        x = self.double_conv4(x)

        if self.concat_input:
            x = self.up_conv_input(x)
            x = torch.cat([x, input_], dim=1)
            x = self.double_conv_input(x)

        x = self.final_conv(x)

        return x

class EfficientUnet_Dual_Decoder(nn.Module):
    def __init__(self, encoder, out_channels=2, concat_input=True):
        super().__init__()

        self.encoder = encoder
        self.concat_input = concat_input


        #
        self.up_conv1 = up_conv(self.n_channels, 512)
        self.double_conv1 = double_conv(self.size[0], 512)
        self.up_conv2 = up_conv(512, 256)
        self.double_conv2 = double_conv(self.size[1], 256)
        self.up_conv3 = up_conv(256, 128)
        self.double_conv3 = double_conv(self.size[2], 128)
        self.up_conv4 = up_conv(128, 64)
        self.double_conv4 = double_conv(self.size[3], 64)

        if self.concat_input:
            self.up_conv_input = up_conv(64, 32)
            self.double_conv_input = double_conv(self.size[4], 32)

        self.final_conv = nn.Conv2d(self.size[5], out_channels, kernel_size=1)


        # aux
        self.up_conv1_aux = up_conv(self.n_channels, 512)
        self.double_conv1_aux = double_conv(self.size[0], 512)
        self.up_conv2_aux = up_conv(512, 256)
        self.double_conv2_aux = double_conv(self.size[1], 256)
        self.up_conv3_aux = up_conv(256, 128)
        self.double_conv3_aux = double_conv(self.size[2], 128)
        self.up_conv4_aux = up_conv(128, 64)
        self.double_conv4_aux = double_conv(self.size[3], 64)

        if self.concat_input:
            self.up_conv_input_aux = up_conv(64, 32)
            self.double_conv_input_aux = double_conv(self.size[4], 32)

        self.final_conv_aux = nn.Conv2d(self.size[5], 2, kernel_size=1)

    @property
    def n_channels(self):
        n_channels_dict = {'efficientnet-b0': 1280, 'efficientnet-b1': 1280, 'efficientnet-b2': 1408,
                           'efficientnet-b3': 1536, 'efficientnet-b4': 1792, 'efficientnet-b5': 2048,
                           'efficientnet-b6': 2304, 'efficientnet-b7': 2560}
        return n_channels_dict[self.encoder.name]

    @property
    def size(self):
        size_dict = {'efficientnet-b0': [592, 296, 152, 80, 35, 32], 'efficientnet-b1': [592, 296, 152, 80, 35, 32],
                     'efficientnet-b2': [600, 304, 152, 80, 35, 32], 'efficientnet-b3': [608, 304, 160, 88, 35, 32],
                     'efficientnet-b4': [624, 312, 160, 88, 35, 32], 'efficientnet-b5': [640, 320, 168, 88, 35, 32],
                     'efficientnet-b6': [656, 328, 168, 96, 35, 32], 'efficientnet-b7': [672, 336, 176, 96, 35, 32]}
        return size_dict[self.encoder.name]

    def forward(self, x):
        input_ = x

        blocks = get_blocks_to_be_concat(self.encoder, x)
        _, x0 = blocks.popitem()

        x = self.up_conv1(x0)
        x1 = blocks.popitem()[1]
        x = torch.cat([x, x1], dim=1)
        x = self.double_conv1(x)
        x_aux = self.up_conv1_aux(x0)
        x_aux = torch.cat([x_aux, x1], dim=1)
        x_aux = self.double_conv1_aux(x_aux)

        x = self.up_conv2(x)
        x2 = blocks.popitem()[1]
        x = torch.cat([x, x2], dim=1)
        x = self.double_conv2(x)
        x_aux = self.up_conv2_aux(x_aux)
        x_aux = torch.cat([x_aux, x2], dim=1)
        x_aux = self.double_conv2_aux(x_aux)

        x = self.up_conv3(x)
        x3 = blocks.popitem()[1]
        x = torch.cat([x, x3], dim=1)
        x = self.double_conv3(x)
        x_aux = self.up_conv3_aux(x_aux)
        x_aux = torch.cat([x_aux, x3], dim=1)
        x_aux = self.double_conv3_aux(x_aux)

        x = self.up_conv4(x)
        x4 = blocks.popitem()[1]
        x = torch.cat([x, x4], dim=1)
        x = self.double_conv4(x)
        x_aux = self.up_conv4_aux(x_aux)
        x_aux = torch.cat([x_aux, x4], dim=1)
        x_aux = self.double_conv4_aux(x_aux)

        if self.concat_input:
            x = self.up_conv_input(x)
            x = torch.cat([x, input_], dim=1)
            x = self.double_conv_input(x)
            x_aux = self.up_conv_input_aux(x_aux)
            x_aux = torch.cat([x_aux, input_], dim=1)
            x_aux = self.double_conv_input_aux(x_aux)

        x = self.final_conv(x)
        x_aux = self.final_conv_aux(x_aux)

        return x, x_aux


class EfficientUnet_SR(nn.Module):
    def __init__(self, encoder, out_channels=2, concat_input=True):
        super().__init__()

        self.encoder = encoder
        self.concat_input = concat_input
        self.bakcbone_channels = [16,24,40,80]
        ####SR
        self.gap = nn.Sequential(
            nn.Conv2d(1280,self.bakcbone_channels[-1],1,1),
            GlobalAvgPool2D()
        )
        self.fpn = fpn.FPN(
                            in_channels_list=self.bakcbone_channels,
                            out_channels=self.bakcbone_channels[-1],
                            conv_block=fpn.default_conv_block,
                            top_blocks=None,)
        self.sr = SceneRelation(
                            in_channels=self.bakcbone_channels[-1],
                            channel_list=(self.bakcbone_channels[-1], self.bakcbone_channels[-1], self.bakcbone_channels[-1], self.bakcbone_channels[-1]),
                            out_channels=128,
                            scale_aware_proj=True,
                        )


        self.up_conv1 = up_conv(self.n_channels, 512)
        self.double_conv1 = double_conv(self.size[0], 512)

        # decoder部分
        self.up_conv2 = up_conv(512, 256)
        self.double_conv2 = double_conv(self.size[1], 256)
        self.up_conv3 = up_conv(256, 128)
        self.double_conv3 = double_conv(self.size[2], 128)
        self.up_conv4 = up_conv(128, 64)
        self.double_conv4 = double_conv(self.size[3], 64)

        if self.concat_input:
            self.up_conv_input = up_conv(64, 32)
            self.double_conv_input = double_conv(self.size[4], 32)

        self.final_conv = nn.Conv2d(self.size[5], out_channels, kernel_size=1)




    @property
    def n_channels(self):
        n_channels_dict = {'efficientnet-b0': 1280, 'efficientnet-b1': 1280, 'efficientnet-b2': 1408,
                           'efficientnet-b3': 1536, 'efficientnet-b4': 1792, 'efficientnet-b5': 2048,
                           'efficientnet-b6': 2304, 'efficientnet-b7': 2560}
        return n_channels_dict[self.encoder.name]

    @property
    def size(self):
        size_dict = {'efficientnet-b0': [592 + 48, 296+88, 152+104, 80+112, 35, 32], 'efficientnet-b1': [592, 296, 152, 80, 35, 32],
                     'efficientnet-b2': [600, 304, 152, 80, 35, 32], 'efficientnet-b3': [608, 304, 160, 88, 35, 32],
                     'efficientnet-b4': [624, 312, 160, 88, 35, 32], 'efficientnet-b5': [640, 320, 168, 88, 35, 32],
                     'efficientnet-b6': [656, 328, 168, 96, 35, 32], 'efficientnet-b7': [672, 336, 176, 96, 35, 32]}
        return size_dict[self.encoder.name]

    def forward(self, x):
        input_ = x

        blocks = get_blocks_to_be_concat(self.encoder, x)
        _, x = blocks.popitem()

        c6 = self.gap(x)
        fpn_feature = self.fpn(list(blocks.values()))
        refined_fpn_feature = self.sr(c6, fpn_feature)
        blocks = {str(index): value for index, value in enumerate(refined_fpn_feature)}

        x = self.up_conv1(x)
        x = torch.cat([x, blocks.popitem()[1]], dim=1)
        x = self.double_conv1(x)

        x = self.up_conv2(x)
        x = torch.cat([x, blocks.popitem()[1]], dim=1)
        x = self.double_conv2(x)

        x = self.up_conv3(x)
        x = torch.cat([x, blocks.popitem()[1]], dim=1)
        x = self.double_conv3(x)

        x = self.up_conv4(x)
        x = torch.cat([x, blocks.popitem()[1]], dim=1)
        x = self.double_conv4(x)

        if self.concat_input:
            x = self.up_conv_input(x)
            x = torch.cat([x, input_], dim=1)
            x = self.double_conv_input(x)

        x = self.final_conv(x)

        return x


def get_efficientunet_b0(out_channels=2, concat_input=True, pretrained=True):
    encoder = EfficientNet.encoder('efficientnet-b0', pretrained=pretrained)
    model = EfficientUnet(encoder, out_channels=out_channels, concat_input=concat_input)
    return model

def get_efficientunet_SR_b0(out_channels=2, concat_input=True, pretrained=True):
    encoder = EfficientNet.encoder('efficientnet-b0', pretrained=pretrained)
    model = EfficientUnet_SR(encoder, out_channels=out_channels, concat_input=concat_input)
    return model


def get_efficientunet_b1(out_channels=2, concat_input=True, pretrained=True):
    encoder = EfficientNet.encoder('efficientnet-b1', pretrained=pretrained)
    model = EfficientUnet(encoder, out_channels=out_channels, concat_input=concat_input)
    return model


def get_efficientunet_b2(out_channels=2, concat_input=True, pretrained=True):
    encoder = EfficientNet.encoder('efficientnet-b2', pretrained=pretrained)
    model = EfficientUnet(encoder, out_channels=out_channels, concat_input=concat_input)
    return model


def get_efficientunet_b3(out_channels=2, concat_input=True, pretrained=True):
    encoder = EfficientNet.encoder('efficientnet-b3', pretrained=pretrained)
    model = EfficientUnet(encoder, out_channels=out_channels, concat_input=concat_input)
    return model


def get_efficientunet_b4(out_channels=2, concat_input=True, pretrained=True):
    encoder = EfficientNet.encoder('efficientnet-b4', pretrained=pretrained)
    model = EfficientUnet(encoder, out_channels=out_channels, concat_input=concat_input)
    return model


def get_efficientunet_b5(out_channels=2, concat_input=True, pretrained=True):
    encoder = EfficientNet.encoder('efficientnet-b5', pretrained=pretrained)
    model = EfficientUnet(encoder, out_channels=out_channels, concat_input=concat_input)
    return model


def get_efficientunet_b6(out_channels=2, concat_input=True, pretrained=True):
    encoder = EfficientNet.encoder('efficientnet-b6', pretrained=pretrained)
    model = EfficientUnet(encoder, out_channels=out_channels, concat_input=concat_input)
    return model


def get_efficientunet_b7(out_channels=2, concat_input=True, pretrained=True):
    encoder = EfficientNet.encoder('efficientnet-b7', pretrained=pretrained)
    model = EfficientUnet(encoder, out_channels=out_channels, concat_input=concat_input)
    return model


def get_efficientunet(num_classes=2,backbone='b0',pretrained=True):
    if backbone == 'b0':
        return get_efficientunet_b0(out_channels=num_classes,pretrained=pretrained)
    if backbone == 'b1':
        return get_efficientunet_b1(out_channels=num_classes,pretrained=pretrained)

def get_efficientunet_aux(num_classes=2,backbone='b0',pretrained=True):
    if backbone == 'b0':
        encoder = EfficientNet.encoder('efficientnet-b0', pretrained=pretrained)
        model = EfficientUnet_Dual_Decoder(encoder, out_channels=num_classes, concat_input=True)
    if backbone == 'b1':
        encoder = EfficientNet.encoder('efficientnet-b1', pretrained=pretrained)
        model = EfficientUnet_Dual_Decoder(encoder, out_channels=num_classes, concat_input=True)
    return model


def get_efficientunet_SR(num_classes=2,backbone='b0',pretrained=True):
    if backbone == 'b0':
        return get_efficientunet_SR_b0(out_channels=num_classes,pretrained=pretrained)
    if backbone == 'b1':
        return get_efficientunet_b1(out_channels=num_classes,pretrained=pretrained)


if __name__ == '__main__':
    data = torch.randn(4,3,256,256)
    # model = get_efficientunet_SR_b0(out_channels=2,concat_input=True,pretrained=False)
    # out = model(data)

    encoder = EfficientNet.encoder('efficientnet-b0', pretrained=False)
    # model = EfficientUnet_SR(encoder, out_channels=out_channels, concat_input=concat_input)
    model = EfficientUnet_Dual_Decoder(encoder,out_channels=5,concat_input=True)
    out,out_aux = model(data)
    print(out.shape)
    print(out_aux.shape)