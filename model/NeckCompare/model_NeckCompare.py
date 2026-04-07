import torch
import torch.nn as nn
import torch.nn.functional as F

from model.ACM.fusion import AsymBiChaFuse


AVAILABLE_NECKS = ("spp", "fpn", "panet", "acm")


def resize_to(feature, reference):
    if feature.shape[-2:] == reference.shape[-2:]:
        return feature
    return F.interpolate(feature, size=reference.shape[-2:], mode="bilinear", align_corners=False)


class ConvBNAct(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=None, activation=True):
        if padding is None:
            padding = kernel_size // 2

        layers = [
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
        ]
        if activation:
            layers.append(nn.ReLU(inplace=True))
        super().__init__(*layers)


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = ConvBNAct(in_channels, out_channels, kernel_size=3, stride=stride)
        self.conv2 = ConvBNAct(out_channels, out_channels, kernel_size=3, activation=False)
        if stride != 1 or in_channels != out_channels:
            self.shortcut = ConvBNAct(
                in_channels,
                out_channels,
                kernel_size=1,
                stride=stride,
                padding=0,
                activation=False,
            )
        else:
            self.shortcut = nn.Identity()
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        residual = self.shortcut(x)
        x = self.conv1(x)
        x = self.conv2(x)
        return self.relu(x + residual)


class ResidualStage(nn.Sequential):
    def __init__(self, in_channels, out_channels, num_blocks=1, stride=1):
        layers = [ResidualBlock(in_channels, out_channels, stride=stride)]
        for _ in range(num_blocks - 1):
            layers.append(ResidualBlock(out_channels, out_channels))
        super().__init__(*layers)


class LightweightResidualEncoder(nn.Module):
    def __init__(self, in_channels=1, channels=(16, 32, 64, 128), blocks_per_stage=(1, 1, 1, 1)):
        super().__init__()
        self.stem = nn.Sequential(
            ConvBNAct(in_channels, channels[0], kernel_size=3),
            ResidualBlock(channels[0], channels[0]),
        )
        self.stage1 = ResidualStage(channels[0], channels[0], num_blocks=blocks_per_stage[0], stride=1)
        self.stage2 = ResidualStage(channels[0], channels[1], num_blocks=blocks_per_stage[1], stride=2)
        self.stage3 = ResidualStage(channels[1], channels[2], num_blocks=blocks_per_stage[2], stride=2)
        self.stage4 = ResidualStage(channels[2], channels[3], num_blocks=blocks_per_stage[3], stride=2)

    def forward(self, x):
        c1 = self.stage1(self.stem(x))
        c2 = self.stage2(c1)
        c3 = self.stage3(c2)
        c4 = self.stage4(c3)
        return c1, c2, c3, c4


class SegmentationHead(nn.Sequential):
    def __init__(self, in_channels, num_classes):
        super().__init__(
            ConvBNAct(in_channels, in_channels, kernel_size=3),
            nn.Conv2d(in_channels, num_classes, kernel_size=1),
        )


class ConcatFusionBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.block = nn.Sequential(
            ConvBNAct(channels * 2, channels, kernel_size=3),
            ResidualBlock(channels, channels),
        )

    def forward(self, high_feature, low_feature):
        high_feature = resize_to(high_feature, low_feature)
        return self.block(torch.cat([low_feature, high_feature], dim=1))


class SPPContextBlock(nn.Module):
    def __init__(self, in_channels, out_channels, pool_sizes=(1, 2, 4)):
        super().__init__()
        branch_channels = max(out_channels // len(pool_sizes), 1)
        self.branches = nn.ModuleList(
            [
                nn.Sequential(
                    nn.AdaptiveAvgPool2d(output_size=pool_size),
                    nn.Conv2d(in_channels, branch_channels, kernel_size=1, bias=False),
                    nn.ReLU(inplace=True),
                )
                for pool_size in pool_sizes
            ]
        )
        merged_channels = in_channels + branch_channels * len(pool_sizes)
        self.project = nn.Sequential(
            ConvBNAct(merged_channels, out_channels, kernel_size=1, padding=0),
            ResidualBlock(out_channels, out_channels),
        )

    def forward(self, x):
        features = [x]
        for branch in self.branches:
            pooled = branch(x)
            features.append(resize_to(pooled, x))
        return self.project(torch.cat(features, dim=1))


class SPPNeck(nn.Module):
    def __init__(self, encoder_channels, neck_channels):
        super().__init__()
        c1, c2, c3, c4 = encoder_channels
        self.lateral1 = ConvBNAct(c1, neck_channels, kernel_size=1, padding=0)
        self.lateral2 = ConvBNAct(c2, neck_channels, kernel_size=1, padding=0)
        self.lateral3 = ConvBNAct(c3, neck_channels, kernel_size=1, padding=0)
        self.context = SPPContextBlock(c4, neck_channels)
        self.fuse3 = ConcatFusionBlock(neck_channels)
        self.fuse2 = ConcatFusionBlock(neck_channels)
        self.fuse1 = ConcatFusionBlock(neck_channels)
        self.out_refine = ResidualBlock(neck_channels, neck_channels)

    def forward(self, features):
        c1, c2, c3, c4 = features
        l1 = self.lateral1(c1)
        l2 = self.lateral2(c2)
        l3 = self.lateral3(c3)
        x = self.context(c4)
        x = self.fuse3(x, l3)
        x = self.fuse2(x, l2)
        x = self.fuse1(x, l1)
        return self.out_refine(x)


class FPNNeck(nn.Module):
    def __init__(self, encoder_channels, neck_channels):
        super().__init__()
        c1, c2, c3, c4 = encoder_channels
        self.laterals = nn.ModuleList(
            [
                ConvBNAct(c1, neck_channels, kernel_size=1, padding=0),
                ConvBNAct(c2, neck_channels, kernel_size=1, padding=0),
                ConvBNAct(c3, neck_channels, kernel_size=1, padding=0),
                ConvBNAct(c4, neck_channels, kernel_size=1, padding=0),
            ]
        )
        self.smooth = nn.ModuleList(
            [ResidualBlock(neck_channels, neck_channels) for _ in range(3)]
        )
        self.aggregate = nn.Sequential(
            ConvBNAct(neck_channels * 4, neck_channels, kernel_size=3),
            ResidualBlock(neck_channels, neck_channels),
        )

    def forward(self, features):
        c1, c2, c3, c4 = features
        p1 = self.laterals[0](c1)
        p2 = self.laterals[1](c2)
        p3 = self.laterals[2](c3)
        p4 = self.laterals[3](c4)

        p3 = self.smooth[2](p3 + resize_to(p4, p3))
        p2 = self.smooth[1](p2 + resize_to(p3, p2))
        p1 = self.smooth[0](p1 + resize_to(p2, p1))

        merged = torch.cat(
            [p1, resize_to(p2, p1), resize_to(p3, p1), resize_to(p4, p1)],
            dim=1,
        )
        return self.aggregate(merged)


class PANetNeck(nn.Module):
    def __init__(self, encoder_channels, neck_channels):
        super().__init__()
        c1, c2, c3, c4 = encoder_channels
        self.laterals = nn.ModuleList(
            [
                ConvBNAct(c1, neck_channels, kernel_size=1, padding=0),
                ConvBNAct(c2, neck_channels, kernel_size=1, padding=0),
                ConvBNAct(c3, neck_channels, kernel_size=1, padding=0),
                ConvBNAct(c4, neck_channels, kernel_size=1, padding=0),
            ]
        )
        self.top_down = nn.ModuleList(
            [ResidualBlock(neck_channels, neck_channels) for _ in range(3)]
        )
        self.downsample = nn.ModuleList(
            [ConvBNAct(neck_channels, neck_channels, kernel_size=3, stride=2) for _ in range(3)]
        )
        self.bottom_up = nn.ModuleList(
            [ResidualBlock(neck_channels, neck_channels) for _ in range(3)]
        )
        self.aggregate = nn.Sequential(
            ConvBNAct(neck_channels * 4, neck_channels, kernel_size=3),
            ResidualBlock(neck_channels, neck_channels),
        )

    def forward(self, features):
        c1, c2, c3, c4 = features
        p1 = self.laterals[0](c1)
        p2 = self.laterals[1](c2)
        p3 = self.laterals[2](c3)
        p4 = self.laterals[3](c4)

        p3 = self.top_down[2](p3 + resize_to(p4, p3))
        p2 = self.top_down[1](p2 + resize_to(p3, p2))
        p1 = self.top_down[0](p1 + resize_to(p2, p1))

        n1 = p1
        n2 = self.bottom_up[0](p2 + resize_to(self.downsample[0](n1), p2))
        n3 = self.bottom_up[1](p3 + resize_to(self.downsample[1](n2), p3))
        n4 = self.bottom_up[2](p4 + resize_to(self.downsample[2](n3), p4))

        merged = torch.cat(
            [n1, resize_to(n2, n1), resize_to(n3, n1), resize_to(n4, n1)],
            dim=1,
        )
        return self.aggregate(merged)


class ACMNeck(nn.Module):
    def __init__(self, encoder_channels, neck_channels):
        super().__init__()
        c1, c2, c3, c4 = encoder_channels
        self.laterals = nn.ModuleList(
            [
                ConvBNAct(c1, neck_channels, kernel_size=1, padding=0),
                ConvBNAct(c2, neck_channels, kernel_size=1, padding=0),
                ConvBNAct(c3, neck_channels, kernel_size=1, padding=0),
                ConvBNAct(c4, neck_channels, kernel_size=1, padding=0),
            ]
        )
        self.fuse3 = AsymBiChaFuse(channels=neck_channels)
        self.fuse2 = AsymBiChaFuse(channels=neck_channels)
        self.fuse1 = AsymBiChaFuse(channels=neck_channels)
        self.refine3 = ResidualBlock(neck_channels, neck_channels)
        self.refine2 = ResidualBlock(neck_channels, neck_channels)
        self.refine1 = ResidualBlock(neck_channels, neck_channels)
        self.aggregate = nn.Sequential(
            ConvBNAct(neck_channels * 4, neck_channels, kernel_size=3),
            ResidualBlock(neck_channels, neck_channels),
        )

    def forward(self, features):
        c1, c2, c3, c4 = features
        p1 = self.laterals[0](c1)
        p2 = self.laterals[1](c2)
        p3 = self.laterals[2](c3)
        p4 = self.laterals[3](c4)

        p3 = self.refine3(self.fuse3(resize_to(p4, p3), p3))
        p2 = self.refine2(self.fuse2(resize_to(p3, p2), p2))
        p1 = self.refine1(self.fuse1(resize_to(p2, p1), p1))

        merged = torch.cat(
            [p1, resize_to(p2, p1), resize_to(p3, p1), resize_to(p4, p1)],
            dim=1,
        )
        return self.aggregate(merged)


def build_neck(neck_type, encoder_channels, neck_channels):
    neck_type = neck_type.lower()
    if neck_type == "spp":
        return SPPNeck(encoder_channels, neck_channels)
    if neck_type == "fpn":
        return FPNNeck(encoder_channels, neck_channels)
    if neck_type == "panet":
        return PANetNeck(encoder_channels, neck_channels)
    if neck_type == "acm":
        return ACMNeck(encoder_channels, neck_channels)
    raise ValueError("Unknown neck_type '{}'. Supported necks: {}.".format(neck_type, ", ".join(AVAILABLE_NECKS)))


class ModularIRSTDNet(nn.Module):
    """Shared lightweight backbone with a pluggable neck for fast ablations."""

    def __init__(
        self,
        in_channels=1,
        num_classes=1,
        neck_type="spp",
        encoder_channels=(16, 32, 64, 128),
        neck_channels=32,
        blocks_per_stage=(1, 1, 1, 1),
    ):
        super().__init__()
        self.neck_type = neck_type.lower()
        self.encoder = LightweightResidualEncoder(
            in_channels=in_channels,
            channels=encoder_channels,
            blocks_per_stage=blocks_per_stage,
        )
        self.neck = build_neck(self.neck_type, encoder_channels, neck_channels)
        self.head = SegmentationHead(neck_channels, num_classes)

    def forward(self, x):
        features = self.encoder(x)
        fused = self.neck(features)
        logits = self.head(fused)
        logits = resize_to(logits, x)
        return torch.sigmoid(logits)

    def evaluate(self, x):
        return self.forward(x)


class CompareSPP(ModularIRSTDNet):
    def __init__(self, **kwargs):
        super().__init__(neck_type="spp", **kwargs)


class CompareFPN(ModularIRSTDNet):
    def __init__(self, **kwargs):
        super().__init__(neck_type="fpn", **kwargs)


class ComparePANet(ModularIRSTDNet):
    def __init__(self, **kwargs):
        super().__init__(neck_type="panet", **kwargs)


class CompareACM(ModularIRSTDNet):
    def __init__(self, **kwargs):
        super().__init__(neck_type="acm", **kwargs)
