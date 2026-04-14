"""
NonLocal注意力模块实现
用于定位关键区域，解决遮挡问题
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from src.core.registry import MODEL_REGISTRY


@MODEL_REGISTRY.register()
class NonLocalBlock(nn.Module):
    """
    NonLocal Neural Networks 注意力模块
    
    通过计算特征图中所有位置之间的相关性，捕获长距离依赖关系，
    有效解决遮挡问题。
    
    Reference:
        "Non-local Neural Networks" (CVPR 2018)
    
    Args:
        in_channels: 输入通道数
        inter_channels: 中间通道数，默认为in_channels//2
        sub_sample: 是否对特征图进行下采样以减少计算量
        bn_layer: 是否使用BatchNorm
    """
    
    def __init__(self, in_channels: int, inter_channels: int = None,
                 sub_sample: bool = True, bn_layer: bool = True):
        super(NonLocalBlock, self).__init__()
        
        self.in_channels = in_channels
        self.inter_channels = inter_channels or in_channels // 2
        
        # 确保inter_channels不为0
        if self.inter_channels == 0:
            self.inter_channels = 1
        
        # 定义卷积层
        self.g = nn.Conv2d(in_channels, self.inter_channels, kernel_size=1)
        self.theta = nn.Conv2d(in_channels, self.inter_channels, kernel_size=1)
        self.phi = nn.Conv2d(in_channels, self.inter_channels, kernel_size=1)
        
        # 输出卷积
        if bn_layer:
            self.W = nn.Sequential(
                nn.Conv2d(self.inter_channels, in_channels, kernel_size=1),
                nn.BatchNorm2d(in_channels)
            )
            # 初始化BatchNorm为0，使模块初始时为单位映射
            nn.init.constant_(self.W[1].weight, 0)
            nn.init.constant_(self.W[1].bias, 0)
        else:
            self.W = nn.Conv2d(self.inter_channels, in_channels, kernel_size=1)
            nn.init.constant_(self.W.weight, 0)
            nn.init.constant_(self.W.bias, 0)
        
        # 下采样减少计算量
        if sub_sample:
            self.g = nn.Sequential(
                self.g,
                nn.MaxPool2d(kernel_size=2)
            )
            self.phi = nn.Sequential(
                self.phi,
                nn.MaxPool2d(kernel_size=2)
            )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            x: 输入特征图 [B, C, H, W]
            
        Returns:
            输出特征图 [B, C, H, W]
        """
        batch_size, C, H, W = x.size()
        
        # g(x) - 用于加权聚合
        g_x = self.g(x).view(batch_size, self.inter_channels, -1)
        g_x = g_x.permute(0, 2, 1)  # [B, H'*W', C//2]
        
        # theta(x) - 查询
        theta_x = self.theta(x).view(batch_size, self.inter_channels, -1)
        theta_x = theta_x.permute(0, 2, 1)  # [B, H*W, C//2]
        
        # phi(x) - 键
        phi_x = self.phi(x).view(batch_size, self.inter_channels, -1)  # [B, C//2, H'*W']
        
        # 计算注意力权重: f = theta(x) * phi(x)^T
        f = torch.matmul(theta_x, phi_x)  # [B, H*W, H'*W']
        f_div_C = F.softmax(f, dim=-1)  # 归一化
        
        # 加权聚合: y = f * g(x)
        y = torch.matmul(f_div_C, g_x)  # [B, H*W, C//2]
        y = y.permute(0, 2, 1).contiguous()
        y = y.view(batch_size, self.inter_channels, H, W)
        
        # 输出变换
        W_y = self.W(y)
        
        # 残差连接
        z = W_y + x
        
        return z


class EmbeddedGaussianNonLocal(NonLocalBlock):
    """Embedded Gaussian版本的NonLocal"""
    pass  # 默认实现就是Embedded Gaussian


class GaussianNonLocal(NonLocalBlock):
    """Gaussian版本的NonLocal，不使用卷积变换"""
    
    def __init__(self, in_channels: int, sub_sample: bool = True):
        super().__init__(in_channels, in_channels, sub_sample, bn_layer=False)
        # 移除卷积层，直接使用特征
        self.theta = nn.Identity()
        self.phi = nn.Identity()
        self.g = nn.Identity()
        self.inter_channels = in_channels


class DotProductNonLocal(NonLocalBlock):
    """Dot Product版本的NonLocal，不使用softmax"""
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, C, H, W = x.size()
        
        g_x = self.g(x).view(batch_size, self.inter_channels, -1)
        g_x = g_x.permute(0, 2, 1)
        
        theta_x = self.theta(x).view(batch_size, self.inter_channels, -1)
        theta_x = theta_x.permute(0, 2, 1)
        
        phi_x = self.phi(x).view(batch_size, self.inter_channels, -1)
        
        f = torch.matmul(theta_x, phi_x)
        # 不使用softmax，而是除以N进行归一化
        N = f.size(-1)
        f_div_C = f / N
        
        y = torch.matmul(f_div_C, g_x)
        y = y.permute(0, 2, 1).contiguous()
        y = y.view(batch_size, self.inter_channels, H, W)
        
        W_y = self.W(y)
        z = W_y + x
        
        return z


if __name__ == "__main__":
    # 测试
    x = torch.randn(2, 512, 28, 28)
    nl = NonLocalBlock(512)
    out = nl(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {out.shape}")
