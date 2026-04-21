"""
Model 2 — İyileştirilmiş LeNet-5 (BatchNorm + Dropout)
=======================================================

Ödev gereksinimi (PDF): "İkinci model için de bir CNN sınıfı açık olarak
yazılacaktır. Bu CNN sınıfında ilk sınıftaki katmanların hiperparametreleri
aynı tutulacak olup ağı iyileştirmesi beklenen özel katmanlar ... eklenebilir.
Bu katmanlar batch normalizasyon katmanları ya da dropout katmanları olabilir."

Bu nedenle:
    - Tüm Conv2d (kanal, kernel) ve Linear (in_features, out_features)
      hiperparametreleri Model 1 (LeNet5Base) ile **birebir aynıdır**.
    - Her evrişim sonrası BatchNorm2d eklenmiştir (aktivasyondan önce).
    - Her fc katmanından ÖNCE Dropout(p=0.5) eklenmiştir.
    - nn.Sequential kullanılmadı: her katman __init__'te ayrı, forward'da
      tek tek çağrılır ("açık sınıf" gereksinimi).

Mimari (Input: [B, 1, 32, 32] — MNIST + Pad(2)):

    Katman                                         Çıktı şekli
    ---------------------------------------------  -----------------
    conv1  : Conv2d(1, 6, k=5)            (AYNI)   [B, 6, 28, 28]
    bn1    : BatchNorm2d(6)               (YENİ)   [B, 6, 28, 28]
    ReLU
    pool1  : MaxPool2d(k=2, s=2)          (AYNI)   [B, 6, 14, 14]
    conv2  : Conv2d(6, 16, k=5)           (AYNI)   [B, 16, 10, 10]
    bn2    : BatchNorm2d(16)              (YENİ)   [B, 16, 10, 10]
    ReLU
    pool2  : MaxPool2d(k=2, s=2)          (AYNI)   [B, 16, 5, 5]
    conv3  : Conv2d(16, 120, k=5)         (AYNI)   [B, 120, 1, 1]
    bn3    : BatchNorm2d(120)             (YENİ)   [B, 120, 1, 1]
    ReLU
    flatten: view(B, -1)                           [B, 120]
    dropout1: Dropout(p=0.5)              (YENİ)   [B, 120]
    fc1    : Linear(120, 84)              (AYNI)   [B, 84]
    ReLU
    dropout2: Dropout(p=0.5)              (YENİ)   [B, 84]
    fc2    : Linear(84, 10)               (AYNI)   [B, 10]   (logits)

Not: Son katmanda softmax yok (CrossEntropyLoss ham logit bekler).
"""

import torch
import torch.nn as nn


class LeNet5Improved(nn.Module):
    """LeNet-5 + BatchNorm2d + Dropout — MNIST için."""

    def __init__(self) -> None:
        super().__init__()

        # Evrişim katmanları (Model 1 ile AYNI)
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)
        self.conv3 = nn.Conv2d(16, 120, kernel_size=5)

        # YENİ: Batch normalizasyon (her evrişim sonrası)
        self.bn1 = nn.BatchNorm2d(6)
        self.bn2 = nn.BatchNorm2d(16)
        self.bn3 = nn.BatchNorm2d(120)

        # Havuzlama katmanları (Model 1 ile AYNI)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Tam bağlantılı katmanlar (Model 1 ile AYNI)
        self.fc1 = nn.Linear(120, 84)
        self.fc2 = nn.Linear(84, 10)

        # YENİ: Dropout (her fc'den ÖNCE)
        self.dropout1 = nn.Dropout(p=0.5)
        self.dropout2 = nn.Dropout(p=0.5)

        # Aktivasyon
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # C1 + BN + ReLU + Pool1
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.pool1(x)

        # C2 + BN + ReLU + Pool2
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.pool2(x)

        # C3 + BN + ReLU
        x = self.relu(self.bn3(self.conv3(x)))

        # Flatten
        x = x.view(x.size(0), -1)

        # Dropout → FC1 + ReLU → Dropout → FC2
        x = self.dropout1(x)
        x = self.relu(self.fc1(x))
        x = self.dropout2(x)
        x = self.fc2(x)  # ham logit
        return x


if __name__ == "__main__":
    # Sanity-check
    model = LeNet5Improved()
    dummy = torch.randn(4, 1, 32, 32)
    out = model(dummy)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Input : {tuple(dummy.shape)}")
    print(f"Output: {tuple(out.shape)}")
    print(f"Toplam parametre sayısı: {n_params:,}")
