from dynamic_network_architectures.architectures.unet import PlainConvUNet, ResidualEncoderUNet
from dynamic_network_architectures.building_blocks.helper import get_matching_instancenorm, convert_dim_to_conv_op
from dynamic_network_architectures.initialization.weight_init import init_last_bn_before_add_to_0
import torch.nn as nn

class InitWeights_He(object):
    def __init__(self, neg_slope=1e-2):
        self.neg_slope = neg_slope

    def __call__(self, module):
        if isinstance(module, nn.Conv3d) or isinstance(module, nn.Conv2d) or isinstance(module, nn.ConvTranspose2d) or isinstance(module, nn.ConvTranspose3d):
            module.weight = nn.init.kaiming_normal_(module.weight, a=self.neg_slope)
            if module.bias is not None:
                module.bias = nn.init.constant_(module.bias, 0)

def get_network(segmentation_network_class_name: str = "PlainConvUNet",
        num_input_channels: int = 4,
        num_stages: int = 6,
        deep_supervision: bool = True):
    dim = 3
    conv_op = convert_dim_to_conv_op(dim)
    mapping = {
        'PlainConvUNet': PlainConvUNet,
        'ResidualEncoderUNet': ResidualEncoderUNet
    }
    kwargs = {
        'PlainConvUNet': {
            'conv_bias': True,
            'norm_op': get_matching_instancenorm(conv_op),
            'norm_op_kwargs': {'eps': 1e-5, 'affine': True},
            'dropout_op': None, 'dropout_op_kwargs': None,
            'nonlin': nn.LeakyReLU, 'nonlin_kwargs': {'inplace': True},
        },
        'ResidualEncoderUNet': {
            'conv_bias': True,
            'norm_op': get_matching_instancenorm(conv_op),
            'norm_op_kwargs': {'eps': 1e-5, 'affine': True},
            'dropout_op': None, 'dropout_op_kwargs': None,
            'nonlin': nn.LeakyReLU, 'nonlin_kwargs': {'inplace': True},
        }
    }
    assert segmentation_network_class_name in mapping.keys(), 'The network architecture specified by the plans file ' \
                                                              'is non-standard (maybe your own?). Yo\'ll have to dive ' \
                                                              'into either this ' \
                                                              'function (get_network_from_plans) or ' \
                                                              'the init of your nnUNetModule to accommodate that.'
    network_class = mapping[segmentation_network_class_name]
    # conv_or_blocks_per_stage = {
    #     'n_conv_per_stage'
    #     if network_class != ResidualEncoderUNet else 'n_blocks_per_stage': configuration_manager.n_conv_per_stage_encoder,
    #     'n_conv_per_stage_decoder': configuration_manager.n_conv_per_stage_decoder
    # }
    # model = network_class(
    #     input_channels=num_input_channels,
    #     n_stages=num_stages,
    #     features_per_stage=[min(configuration_manager.UNet_base_num_features * 2 ** i,
    #                             configuration_manager.unet_max_num_features) for i in range(num_stages)],
    #     conv_op=conv_op,
    #     kernel_sizes=configuration_manager.conv_kernel_sizes,
    #     strides=configuration_manager.pool_op_kernel_sizes,
    #     num_classes=label_manager.num_segmentation_heads,
    #     deep_supervision=deep_supervision,
    #     **conv_or_blocks_per_stage,
    #     **kwargs[segmentation_network_class_name]
    # )
    # model.apply(InitWeights_He(1e-2))
    # if network_class == ResidualEncoderUNet:
    #     model.apply(init_last_bn_before_add_to_0)
    # return model

if __name__ == '__main__':
    import torch
    data = torch.rand((1, 4, 128, 128, 128))
    model = PlainConvUNet(4, 6, (32, 64, 125, 256, 320, 320), nn.Conv3d, 3, (1, 2, 2, 2, 2, 2), (2, 2, 2, 2, 2, 2), 4,
                                (2, 2, 2, 2, 2), False, nn.BatchNorm3d, None, None, None, nn.ReLU, deep_supervision=True)
    preds = model(data)
    print(len(preds))
    print([preds[i].shape for i in range(len(preds))])
    print(model.compute_conv_feature_map_size(data.shape[2:]))

    # import hiddenlayer as hl
    # g = hl.build_graph(model, data, transforms=None)
    # g.save("network_architecture.pdf")
    # del g