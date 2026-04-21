"""
Model 1 — LeNet-5 Tabanlı CNN (AÇIK sınıf, Sequential yok)
===========================================================

Ödev gereksinimi: "CNN sınıfı açık olarak yazılmalıdır" ve "temel
katmanlar" ayrı ayrı kullanılmalıdır (Conv2d, aktivasyon, pooling,
flatten, FC). Bu nedenle tüm katmanlar __init__ içinde ayrı
değişkenler olarak tanımlanır; forward'da sıralı biçimde çağrılır.

Mimari (Input: [B, 1, 32, 32] — MNIST + Pad(2) sonrası):

    Katman                              Çıktı şekli
    ----------------------------------  -----------------
    conv1  : Conv2d(1, 6, k=5)          [B, 6, 28, 28]
    ReLU
    pool1  : MaxPool2d(k=2, s=2)        [B, 6, 14, 14]
    conv2  : Conv2d(6, 16, k=5)         [B, 16, 10, 10]
    ReLU
    pool2  : MaxPool2d(k=2, s=2)        [B, 16, 5, 5]
    conv3  : Conv2d(16, 120, k=5)       [B, 120, 1, 1]
    ReLU
    flatten: view(B, -1)                [B, 120]
    fc1    : Linear(120, 84)            [B, 84]
    ReLU
    fc2    : Linear(84, 10)             [B, 10]   (logits)

Not: CrossEntropyLoss ham logit bekler; modelin son katmanında
softmax/log-softmax uygulanmaz.
"""

import torch
import torch.nn as nn


class LeNet5Base(nn.Module):
    """LeNet-5 (klasik, residual YOK) — MNIST için."""

    def __init__(self) -> None:
        super().__init__()
        # Evrişim katmanları
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)
        self.conv3 = nn.Conv2d(16, 120, kernel_size=5)

        # Havuzlama katmanları
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Tam bağlantılı katmanlar
        self.fc1 = nn.Linear(120, 84)
        self.fc2 = nn.Linear(84, 10)

        # Aktivasyon
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # C1 + Pool1: [B,1,32,32] → [B,6,28,28] → [B,6,14,14]
        x = self.relu(self.conv1(x))
        x = self.pool1(x)

        # C2 + Pool2: [B,6,14,14] → [B,16,10,10] → [B,16,5,5]
        x = self.relu(self.conv2(x))
        x = self.pool2(x)

        # C3: [B,16,5,5] → [B,120,1,1]
        x = self.relu(self.conv3(x))

        # Flatten: [B,120,1,1] → [B,120]
        x = x.view(x.size(0), -1)

        # FC1: [B,120] → [B,84]
        x = self.relu(self.fc1(x))

        # FC2: [B,84] → [B,10]  (ham logit, softmax YOK)
        x = self.fc2(x)
        return x


if __name__ == "__main__":
    # Sanity-check: tek batch forward + parametre sayımı
    model = LeNet5Base()
    dummy = torch.randn(4, 1, 32, 32)
    out = model(dummy)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Input : {tuple(dummy.shape)}")
    print(f"Output: {tuple(out.shape)}")
    print(f"Toplam parametre sayısı: {n_params:,}")
