"""
Yardımcı fonksiyonlar.
Aktivasyon fonksiyonları ve ortak kullanılan araçları içerir.
"""

import numpy as np


def sigmoid(Z):
    """Sigmoid aktivasyon fonksiyonu."""
    # Numerik kararlılık için clip
    Z = np.clip(Z, -500, 500)
    return 1.0 / (1.0 + np.exp(-Z))


def sigmoid_derivative(A):
    """Sigmoid türevi. A = sigmoid(Z) olmalı."""
    return A * (1 - A)


def tanh(Z):
    """Tanh aktivasyon fonksiyonu."""
    return np.tanh(Z)


def tanh_derivative(A):
    """Tanh türevi. A = tanh(Z) olmalı."""
    return 1 - np.power(A, 2)


def relu(Z):
    """ReLU aktivasyon fonksiyonu."""
    return np.maximum(0, Z)


def relu_derivative(Z):
    """ReLU türevi."""
    return (Z > 0).astype(float)


def softmax(Z):
    """
    Softmax aktivasyon fonksiyonu.
    Z shape: (n_classes, m) -> çıktı shape: (n_classes, m)
    """
    # Numerik kararlılık
    exp_Z = np.exp(Z - np.max(Z, axis=0, keepdims=True))
    return exp_Z / np.sum(exp_Z, axis=0, keepdims=True)


def one_hot_encode(y, n_classes):
    """
    Etiketleri one-hot encode formatına çevirir.
    
    Parameters
    ----------
    y : np.ndarray, shape (m,) veya (m, 1)
        Sınıf etiketleri (0, 1, ..., n_classes-1)
    n_classes : int
        Toplam sınıf sayısı
    
    Returns
    -------
    np.ndarray, shape (n_classes, m)
        One-hot encode edilmiş etiketler
    """
    y = y.flatten().astype(int)
    m = y.shape[0]
    one_hot = np.zeros((n_classes, m))
    one_hot[y, np.arange(m)] = 1
    return one_hot
