import os
from functools import partial
from collections import OrderedDict
import math
import torch
from torch import nn
from torch.utils import model_zoo
import torch.nn.functional as F


class AbstractModel(nn.Module):
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                m.weight.data = nn.init.kaiming_normal_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def initialize_encoder(self, model, model_url, num_channels_changed=False):
        if os.path.isfile(model_url):
            pretrained_dict = torch.load(model_url)
        else:
            pretrained_dict = model_zoo.load_url(model_url)
        if 'state_dict' in pretrained_dict:
            pretrained_dict = pretrained_dict['state_dict']
            pretrained_dict = {k.replace("module.", ""): v
                               for k, v in pretrained_dict.items()}
        model_dict = model.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items()
                           if k in model_dict}
        if num_channels_changed:
            model.state_dict()[self.first_layer_params_name + '.weight'][
                :, :3, ...] = pretrained_dict[
                    self.first_layer_params_name + '.weight'].data
            skip_layers = [self.first_layer_params_name,
                           self.first_layer_params_name + ".weight"]
            pretrained_dict = {k: v for k, v in pretrained_dict.items() if
                               not any(k.startswith(s) for s in skip_layers)}
        model.load_state_dict(pretrained_dict, strict=False)

    @property
    def first_layer_params_name(self):
        return 'conv1'


class Bottleneck(nn.Module):
    """
    Base class for bottlenecks that implements `forward()` method.
    """

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out = self.se_module(out) + residual
        out = self.relu(out)

        return out


class SS_UnetDecoderBlock(nn.Module):
    def __init__(self, in_channels, middle_channels, out_channels):
        super().__init__()
        self.layer = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.layer(x)


class SEModule(nn.Module):

    def __init__(self, channels, reduction, concat=False):
        super(SEModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(channels, channels // reduction, kernel_size=1,
                             padding=0)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(channels // reduction, channels, kernel_size=1,
                             padding=0)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        module_input = x
        x = self.avg_pool(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        return module_input * x


class SS_ConvBottleneck(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.seq = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, dec, enc):
        x = torch.cat([dec, enc], dim=1)
        return self.seq(x)


class SEResNeXtBottleneck(Bottleneck):
    """
    ResNeXt bottleneck type C with a Squeeze-and-Excitation module.
    """
    expansion = 4

    def __init__(self, inplanes, planes, groups, reduction, stride=1,
                 downsample=None, base_width=4):
        super(SEResNeXtBottleneck, self).__init__()
        width = math.floor(planes * (base_width / 64)) * groups
        self.conv1 = nn.Conv2d(inplanes, width, kernel_size=1, bias=False,
                               stride=1)
        self.bn1 = nn.BatchNorm2d(width)
        self.conv2 = nn.Conv2d(width, width, kernel_size=3, stride=stride,
                               padding=1, groups=groups, bias=False)
        self.bn2 = nn.BatchNorm2d(width)
        self.conv3 = nn.Conv2d(width, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.se_module = SEModule(planes * 4, reduction=reduction)
        self.downsample = downsample
        self.stride = stride


class SS_UNet(AbstractModel):
    def __init__(self, num_classes, num_channels=3, encoder_name='resnet34'):
        if not hasattr(self, 'first_layer_stride_two'):
            self.first_layer_stride_two = False
        if not hasattr(self, 'decoder_block'):
            self.decoder_block = SS_UnetDecoderBlock
        if not hasattr(self, 'bottleneck_type'):
            self.bottleneck_type = SS_ConvBottleneck

        self.filters = ss_encoder_params[encoder_name]['filters']
        self.decoder_filters = ss_encoder_params[encoder_name].get(
            'decoder_filters', self.filters[:-1])
        self.last_upsample_filters = ss_encoder_params[encoder_name].get(
            'last_upsample', self.decoder_filters[0]//2)

        super().__init__()

        self.num_channels = num_channels
        self.num_classes = num_classes

        self.bottlenecks = nn.ModuleList([self.bottleneck_type(
            self.filters[-i - 2] + f, f) for i, f in enumerate(
                reversed(self.decoder_filters[:]))])

        self.decoder_stages = nn.ModuleList(
            [self.get_decoder(idx)
             for idx in range(0, len(self.decoder_filters))])

        if self.first_layer_stride_two:
            self.last_upsample = self.decoder_block(self.decoder_filters[0],
                                                    self.last_upsample_filters,
                                                    self.last_upsample_filters)

        self.final = self.make_final_classifier(
            self.last_upsample_filters if self.first_layer_stride_two else self.decoder_filters[0], num_classes)

        self._initialize_weights()

        encoder = ss_encoder_params[encoder_name]['init_op']()
        self.encoder_stages = nn.ModuleList(
            [self.get_encoder(encoder, idx)
             for idx in range(len(self.filters))])
        if ss_encoder_params[encoder_name]['url'] is not None:
            self.initialize_encoder(encoder,
                                    ss_encoder_params[encoder_name]['url'],
                                    num_channels != 3)

    def forward(self, x):
        x, angles = x
        enc_results = []
        for stage in self.encoder_stages:
            x = stage(x)
            enc_results.append(torch.cat(x, dim=1)
                               if isinstance(x, tuple) else x.clone())
        last_dec_out = enc_results[-1]
        size = last_dec_out.size(2)
        last_dec_out = torch.cat([last_dec_out, F.upsample(angles,
                                                           size=(size, size),
                                                           mode="nearest")],
                                 dim=1)
        x = last_dec_out
        for idx, bottleneck in enumerate(self.bottlenecks):
            rev_idx = - (idx + 1)
            x = self.decoder_stages[rev_idx](x)
            x = bottleneck(x, enc_results[rev_idx - 1])

        if self.first_layer_stride_two:
            x = self.last_upsample(x)

        f = self.final(x)

        return f

    def get_decoder(self, layer):
        in_channels = self.filters[layer + 1] if layer + 1 == len(self.decoder_filters) else self.decoder_filters[layer + 1]
        return self.decoder_block(in_channels,
                                  self.decoder_filters[layer],
                                  self.decoder_filters[max(layer, 0)])

    def make_final_classifier(self, in_filters, num_classes):
        return nn.Sequential(
            nn.Conv2d(in_filters, num_classes, 1, padding=0)
        )

    def get_encoder(self, encoder, layer):
        raise NotImplementedError

    @property
    def first_layer_params(self):
        return _ss_get_layers_params([self.encoder_stages[0]])

    @property
    def layers_except_first_params(self):
        layers = ss_get_slice(
            self.encoder_stages, 1, -1) + [self.bottlenecks,
                                           self.decoder_stages,
                                           self.final]
        return _ss_get_layers_params(layers)


def _ss_get_layers_params(layers):
    return sum((list(l.parameters()) for l in layers), [])


def ss_get_slice(features, start, end):
    if end == -1:
        end = len(features)
    return [features[i] for i in range(start, end)]


def initialize_pretrained_model(model, num_classes, settings):
    assert num_classes == settings['num_classes'], \
        'num_classes should be {}, but is {}'.format(
            settings['num_classes'], num_classes)
    model.load_state_dict(model_zoo.load_url(settings['url']), strict=False)
    model.input_space = settings['input_space']
    model.input_size = settings['input_size']
    model.input_range = settings['input_range']
    model.mean = settings['mean']
    model.std = settings['std']


def se_resnext50_32x4d(num_classes=1000, pretrained='imagenet'):
    model = SENet(SEResNeXtBottleneck, [3, 4, 6, 3], groups=32, reduction=16,
                  dropout_p=None, inplanes=64, input_3x3=False,
                  downsample_kernel_size=1, downsample_padding=0,
                  num_classes=num_classes)
    if pretrained is not None:
        settings = {
            'url': 'http://data.lip6.fr/cadene/pretrainedmodels/se_resnext50_32x4d-a260b3a4.pth',
            'input_space': 'RGB',
            'input_size': [3, 224, 224],
            'input_range': [0, 1],
            'mean': [0.485, 0.456, 0.406],
            'std': [0.229, 0.224, 0.225],
            'num_classes': 1000
            }

        initialize_pretrained_model(model, num_classes, settings)
    return model


class SENet(nn.Module):

    def __init__(self, block, layers, groups, reduction, dropout_p=0.2,
                 inplanes=128, input_3x3=True, downsample_kernel_size=3,
                 downsample_padding=1, num_classes=1000):
        """
        Parameters
        ----------
        block (nn.Module): Bottleneck class.
            - For SENet154: SEBottleneck
            - For SE-ResNet models: SEResNetBottleneck
            - For SE-ResNeXt models:  SEResNeXtBottleneck
        layers (list of ints): Number of residual blocks for 4 layers of the
            network (layer1...layer4).
        groups (int): Number of groups for the 3x3 convolution in each
            bottleneck block.
            - For SENet154: 64
            - For SE-ResNet models: 1
            - For SE-ResNeXt models:  32
        reduction (int): Reduction ratio for Squeeze-and-Excitation modules.
            - For all models: 16
        dropout_p (float or None): Drop probability for the Dropout layer.
            If `None` the Dropout layer is not used.
            - For SENet154: 0.2
            - For SE-ResNet models: None
            - For SE-ResNeXt models: None
        inplanes (int):  Number of input channels for layer1.
            - For SENet154: 128
            - For SE-ResNet models: 64
            - For SE-ResNeXt models: 64
        input_3x3 (bool): If `True`, use three 3x3 convolutions instead of
            a single 7x7 convolution in layer0.
            - For SENet154: True
            - For SE-ResNet models: False
            - For SE-ResNeXt models: False
        downsample_kernel_size (int): Kernel size for downsampling convolutions
            in layer2, layer3 and layer4.
            - For SENet154: 3
            - For SE-ResNet models: 1
            - For SE-ResNeXt models: 1
        downsample_padding (int): Padding for downsampling convolutions in
            layer2, layer3 and layer4.
            - For SENet154: 1
            - For SE-ResNet models: 0
            - For SE-ResNeXt models: 0
        num_classes (int): Number of outputs in `last_linear` layer.
            - For all models: 1000
        """
        super(SENet, self).__init__()
        self.inplanes = inplanes
        if input_3x3:
            layer0_modules = [
                ('conv1', nn.Conv2d(3, 64, 3, stride=2, padding=1,
                                    bias=False)),
                ('bn1', nn.BatchNorm2d(64)),
                ('relu1', nn.ReLU(inplace=True)),
                ('conv2', nn.Conv2d(64, 64, 3, stride=1, padding=1,
                                    bias=False)),
                ('bn2', nn.BatchNorm2d(64)),
                ('relu2', nn.ReLU(inplace=True)),
                ('conv3', nn.Conv2d(64, inplanes, 3, stride=1, padding=1,
                                    bias=False)),
                ('bn3', nn.BatchNorm2d(inplanes)),
                ('relu3', nn.ReLU(inplace=True)),
            ]
        else:
            layer0_modules = [
                ('conv1', nn.Conv2d(3, inplanes, kernel_size=7, stride=2,
                                    padding=3, bias=False)),
                ('bn1', nn.BatchNorm2d(inplanes)),
                ('relu1', nn.ReLU(inplace=True)),
            ]
        # To preserve compatibility with Caffe weights `ceil_mode=True`
        # is used instead of `padding=1`.
        self.pool = nn.MaxPool2d(3, stride=2, ceil_mode=True)
        self.layer0 = nn.Sequential(OrderedDict(layer0_modules))
        self.layer1 = self._make_layer(
            block,
            planes=64,
            blocks=layers[0],
            groups=groups,
            reduction=reduction,
            downsample_kernel_size=1,
            downsample_padding=0
        )
        self.layer2 = self._make_layer(
            block,
            planes=128,
            blocks=layers[1],
            stride=2,
            groups=groups,
            reduction=reduction,
            downsample_kernel_size=downsample_kernel_size,
            downsample_padding=downsample_padding
        )
        self.layer3 = self._make_layer(
            block,
            planes=256,
            blocks=layers[2],
            stride=2,
            groups=groups,
            reduction=reduction,
            downsample_kernel_size=downsample_kernel_size,
            downsample_padding=downsample_padding
        )
        self.layer4 = self._make_layer(
            block,
            planes=512,
            blocks=layers[3],
            stride=2,
            groups=groups,
            reduction=reduction,
            downsample_kernel_size=downsample_kernel_size,
            downsample_padding=downsample_padding
        )
        self.avg_pool = nn.AvgPool2d(7, stride=1)
        self.dropout = nn.Dropout(dropout_p) if dropout_p is not None else None
        self.last_linear = nn.Linear(512 * block.expansion, num_classes)
        self._initialize_weights()

    def _make_layer(self, block, planes, blocks, groups, reduction, stride=1,
                    downsample_kernel_size=1, downsample_padding=0):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=downsample_kernel_size, stride=stride,
                          padding=downsample_padding, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, groups, reduction, stride,
                            downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups, reduction))

        return nn.Sequential(*layers)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                m.weight.data = nn.init.kaiming_normal_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def features(self, x):
        x = self.layer0(x)
        x = self.pool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x

    def logits(self, x):
        x = self.avg_pool(x)
        if self.dropout is not None:
            x = self.dropout(x)
        x = x.view(x.size(0), -1)
        x = self.last_linear(x)
        return x

    def forward(self, x):
        x = self.features(x)
        x = self.logits(x)
        return x


class SCSEModule(nn.Module):
    # according to https://arxiv.org/pdf/1808.08127.pdf concat is better
    def __init__(self, channels, reduction=16, mode='concat'):
        super(SCSEModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(channels, channels // reduction, kernel_size=1,
                             padding=0)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(channels // reduction, channels, kernel_size=1,
                             padding=0)
        self.sigmoid = nn.Sigmoid()

        self.spatial_se = nn.Sequential(nn.Conv2d(channels, 1, kernel_size=1,
                                                  stride=1, padding=0,
                                                  bias=False),
                                        nn.Sigmoid())
        self.mode = mode

    def forward(self, x):
        module_input = x

        x = self.avg_pool(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        chn_se = self.sigmoid(x)
        chn_se = chn_se * module_input

        spa_se = self.spatial_se(module_input)
        spa_se = module_input * spa_se
        if self.mode == 'concat':
            return torch.cat([chn_se, spa_se], dim=1)
        elif self.mode == 'maxout':
            return torch.max(chn_se, spa_se)
        else:
            return chn_se + spa_se


class ConvSCSEBottleneckNoBn(nn.Module):
    def __init__(self, in_channels, out_channels, reduction=2):
        print("bottleneck ", in_channels, out_channels)
        super().__init__()
        self.seq = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.ReLU(inplace=True),
            SCSEModule(out_channels, reduction=reduction, mode='maxout')
        )

    def forward(self, dec, enc):
        x = torch.cat([dec, enc], dim=1)
        return self.seq(x)


ss_encoder_params = {

    # 'dpn92':
    #     {'filters': [64, 336, 704, 1552, 2688 + 27],
    #      'decoder_filters': [64, 128, 256, 256],
    #      'last_upsample': 64,
    #     'init_op': dpn92,
    #     'url': 'http://data.lip6.fr/cadene/pretrainedmodels/dpn92_extra-b040e4a9b.pth'},
    # 'resnet34':
    #     {'filters': [64, 64, 128, 256, 512 + 27],
    #      'decoder_filters': [64, 128, 256, 512],
    #      'last_upsample': 64,
    #      'init_op': partial(resnet.resnet34, in_channels=4),
    #      'url': resnet.model_urls['resnet34']},
    # 'resnet101':
    #     {'filters': [64, 256, 512, 1024, 2048 + 27],
    #      'decoder_filters': [64, 128, 256, 256],
    #      'last_upsample': 64,
    #      'init_op': partial(resnet.resnet101, in_channels=4),
    #      'url': resnet.model_urls['resnet101']},
    # 'densenet121':
    #     {'filters': [64, 256, 512, 1024, 1024 + 27],
    #      'decoder_filters': [64, 128, 256, 256],
    #      'last_upsample': 64,
    #      'url': None,
    #      'init_op': densenet121},
    # 'densenet169':
    #     {'filters': [64, 256, 512, 1280, 1664 + 27],
    #      'decoder_filters': [64, 128, 256, 256],
    #      'last_upsample': 64,
    #      'url': None,
    #      'init_op': densenet169},
    # 'densenet161':
    #     {'filters': [96, 384, 768, 2112, 2208 + 27],
    #      'decoder_filters': [64, 128, 256, 256],
    #      'last_upsample': 64,
    #      'url': None,
    #      'init_op': densenet161},
    'seresnext50':
        {'filters': [64, 256, 512, 1024, 2048 + 27],
         'decoder_filters': [64, 128, 256, 384],
         'init_op': se_resnext50_32x4d,
         'url': 'http://0data.lip6.fr/cadene/pretrainedmodels/se_resnext50_32x4d-a260b3a4.pth'}#,
    # 'senet154':
    #     {'filters': [128, 256, 512, 1024, 2048 + 27],
    #      'decoder_filters': [64, 128, 256, 384],
    #      'init_op': senet154,
    #      'url': 'http://data.lip6.fr/cadene/pretrainedmodels/senet154-c7b49a05.pth'},
    # 'seresnext101':
    #     {'filters': [64, 256, 512, 1024, 2048 + 27],
    #      'decoder_filters': [64, 128, 256, 384],
    #      'last_upsample': 64,
    #      'init_op': se_resnext101_32x4d,
    #      'url': 'http://data.lip6.fr/cadene/pretrainedmodels/se_resnext101_32x4d-3b2fe3d8.pth'},
}


class SelimSef_SpaceNet4_UNetSCSEResNeXt50(SS_UNet):

    def __init__(self, seg_classes):
        self.bottleneck_type = ConvSCSEBottleneckNoBn
        self.first_layer_stride_two = True
        super().__init__(seg_classes, 4, 'seresnext50')

    def get_encoder(self, encoder, layer):
        if layer == 0:
            return encoder.layer0
        elif layer == 1:
            return nn.Sequential(
                encoder.pool,
                encoder.layer1)
        elif layer == 2:
            return encoder.layer2
        elif layer == 3:
            return encoder.layer3
        elif layer == 4:
            return encoder.layer4

    @property
    def first_layer_params_name(self):
        return "layer0.conv1"
