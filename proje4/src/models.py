"""Model factory.

5 model destekleniyor:
    - mlp           : Tam bağlı (fully-connected) baseline. Görüntü yapısını yok sayar.
    - cnn_scratch   : Klasik VGG-mini benzeri CNN, sıfırdan eğitilir (pretrained yok).
    - resnet50      : timm — Klasik CNN, ImageNet pretrained.
    - efficientnet_b0 : timm — Modern CNN, ImageNet pretrained.
    - vit_base      : timm — Vision Transformer, ImageNet pretrained.

İlk ikisi "transfer learning'in değerini" gösteren baseline'lardır.
"""
from __future__ import annotations

import torch.nn as nn

from .config import IMAGE_SIZE, MODEL_CONFIGS, NUM_CLASSES


class MLPClassifier(nn.Module):
    """Yapay Sinir Ağı (ANN/MLP) baseline.

    Görüntüyü düz bir vektöre çevirir (224×224×3 = 150.528 boyut), sonra 2 gizli
    katmanlı tam bağlı ağa verir. Görüntüdeki uzamsal yapıyı (komşuluk, kenar,
    doku) **tamamen yok sayar**. Bu projede transfer learning'in değerini
    göstermek için pedagojik baseline olarak kullanılır.

    Mimari:
        Flatten(150528) → FC 256 → ReLU → Dropout(0.3)
                       → FC 128 → ReLU → Dropout(0.3)
                       → FC 10 (sınıf sayısı)
    """

    def __init__(self, num_classes: int = NUM_CLASSES, image_size: int = IMAGE_SIZE,
                 hidden1: int = 256, hidden2: int = 128, dropout: float = 0.3) -> None:
        super().__init__()
        self.flatten_dim = 3 * image_size * image_size
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.flatten_dim, hidden1),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden1, hidden2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden2, num_classes),
        )

    def forward(self, x):
        return self.net(x)


class CNNScratch(nn.Module):
    """Sıfırdan eğitilen klasik CNN (VGG-mini tarzı).

    4 evrişim bloğu (32 → 64 → 128 → 256 filtre), her blok Conv 3×3 + BatchNorm
    + ReLU + MaxPool 2×2. Sonunda Global Average Pool + 2 katmanlı sınıflandırıcı.
    Pretrained değil — görüntü özelliklerini tamamen kendi veri setimizden
    öğrenmek zorunda. Transfer learning'le karşılaştırma için baseline.

    Mimari:
        [Conv 3×3 → BN → ReLU → MaxPool] × 4 (32, 64, 128, 256 filtre)
        → Global Average Pool
        → FC 128 → ReLU → Dropout(0.3)
        → FC 10
    """

    def __init__(self, num_classes: int = NUM_CLASSES, dropout: float = 0.3) -> None:
        super().__init__()

        def conv_block(in_ch: int, out_ch: int) -> nn.Sequential:
            return nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2, stride=2),
            )

        self.features = nn.Sequential(
            conv_block(3, 32),     # 224 → 112
            conv_block(32, 64),    # 112 → 56
            conv_block(64, 128),   # 56  → 28
            conv_block(128, 256),  # 28  → 14
        )
        self.gap = nn.AdaptiveAvgPool2d(1)   # 256 × 1 × 1
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.gap(x)
        x = self.classifier(x)
        return x


def create_model(name: str, pretrained: bool = True) -> nn.Module:
    if name not in MODEL_CONFIGS:
        raise ValueError(f"Unknown model '{name}'. Choices: {list(MODEL_CONFIGS)}")

    cfg = MODEL_CONFIGS[name]
    kind = cfg.get("kind", "timm")

    if kind == "custom":
        if name == "mlp":
            return MLPClassifier()
        if name == "cnn_scratch":
            return CNNScratch()
        raise ValueError(f"Unknown custom model: {name}")

    # timm tabanlı modeller
    import timm  # local import — timm yoksa proje import edilmeye devam etsin
    model = timm.create_model(
        cfg["timm_name"],
        pretrained=pretrained,
        num_classes=NUM_CLASSES,
    )
    return model


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
