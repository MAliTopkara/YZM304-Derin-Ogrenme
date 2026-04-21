"""
Model 3 / 4 / 5 — ResNet18 Adaptörü (CIFAR-10)
================================================

torchvision.models.resnet18, varsayılan olarak ImageNet (224x224) için
tasarlanmıştır:
    - conv1: 7x7 kernel, stride=2     → 112x112
    - maxpool: 3x3, stride=2          → 56x56

Bu adımlar 32x32 CIFAR görsellerinde feature map'i çok erken küçültür
(doğrudan 4x4'e iner) ve ağın öğrenme kapasitesini zayıflatır.

CIFAR için standart uyarlama:
    - conv1'i 3x3, stride=1, padding=1 yap
    - maxpool'u Identity ile değiştir (kaldır)
    - fc katmanını num_classes çıkışlı yeniden yaz

Aynı adaptör Model 3 (uçtan uca eğitim), Model 4 (feature extraction —
son fc öncesi 512-boyutlu vektör) ve Model 5 (eval-only karşılaştırma)
için kullanılır.
"""

from __future__ import annotations

import torch.nn as nn
from torchvision import models


def get_resnet18_cifar(num_classes: int = 10, pretrained: bool = False) -> nn.Module:
    """
    CIFAR-10 için uyarlanmış ResNet18 döndürür.

    Parameters
    ----------
    num_classes : int
        Son fc katmanının çıktı boyutu.
    pretrained : bool
        True ise torchvision'ın ImageNet ağırlıklarını yükler. Proje
        kapsamında scratch eğitim yapıldığı için varsayılan False.

    Returns
    -------
    nn.Module
        Adapte edilmiş ResNet18 modeli.
    """
    if pretrained:
        weights = models.ResNet18_Weights.DEFAULT
    else:
        weights = None

    model = models.resnet18(weights=weights)

    # conv1: 7x7/s2 → 3x3/s1 (CIFAR 32x32 için)
    model.conv1 = nn.Conv2d(
        in_channels=3,
        out_channels=64,
        kernel_size=3,
        stride=1,
        padding=1,
        bias=False,
    )

    # İlk downsample'i kaldır: 32x32 girdi zaten küçük
    model.maxpool = nn.Identity()

    # Son fc katmanı: num_classes çıkışlı
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)

    return model


if __name__ == "__main__":
    # Sanity-check
    import torch

    model = get_resnet18_cifar(num_classes=10, pretrained=False)
    dummy = torch.randn(4, 3, 32, 32)
    out = model(dummy)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Input : {tuple(dummy.shape)}")
    print(f"Output: {tuple(out.shape)}")
    print(f"Toplam parametre sayısı: {n_params:,}")
    print(f"conv1: {model.conv1}")
    print(f"maxpool: {model.maxpool}")
    print(f"fc: {model.fc}")
