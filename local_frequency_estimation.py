import torch
import torch.nn as nn

class LocalFrequencyEstimationModule(nn.Module):
    def __init__(self,
                 in_channels: int,    # 输入特征维度 C（c_hat 的通道数）
                 latent_dim: int,     # 潜在变量维度（z 的通道数）
                 token_dim: int,      # 高频令牌维度（t 的通道数）
                 kernel_size: int = 3 # 用于预测分布参数的卷积核大小
                 ):
        super().__init__()
        # ----------------------------------------------------------------------------
        # 1) Encoder：预测 2*latent_dim 的参数，分别对应 mu 和 logvar
        # ----------------------------------------------------------------------------
        self.encoder = nn.Conv2d(
            in_channels,
            2 * latent_dim,
            kernel_size=kernel_size,
            padding=kernel_size // 2
        )

        # ----------------------------------------------------------------------------
        # 2) Decoder：把采样后的 z 映射为高频令牌 t
        #    这里用 1×1 卷积当作简单的 MLP
        # ----------------------------------------------------------------------------
        self.decoder = nn.Conv2d(
            latent_dim,
            token_dim,
            kernel_size=1
        )

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """
        依据 mu, logvar 做 reparameterization trick：
        z = mu + sigma * eps, 其中 eps ~ N(0, I)
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, c_hat: torch.Tensor):
        """
        输入：
            c_hat: [B, C, H, W] —— 来自主干网络 unfold／插值后的潜在特征
        返回：
            t_hat: [B, token_dim, H, W] —— 高频令牌
            mu   : [B, latent_dim, H, W]
            logvar: [B, latent_dim, H, W]
        """
        # 1) 预测分布参数
        params = self.encoder(c_hat)              # [B, 2*latent_dim, H, W]
        mu, logvar = torch.chunk(params, 2, dim=1)  # 各自 [B, latent_dim, H, W]

        # 2) 采样
        z = self.reparameterize(mu, logvar)       # [B, latent_dim, H, W]

        # 3) 解码为高频令牌
        t_hat = self.decoder(z)                   # [B, token_dim, H, W]

        return t_hat, mu, logvar
