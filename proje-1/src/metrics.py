"""
Metrik hesaplama ve görselleştirme modülü.
Confusion matrix, classification report, cost-epoch grafikleri.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)


def print_metrics(y_true, y_pred, model_name="Model", task='binary'):
    """
    Sınıflandırma metriklerini yazdırır.
    
    Parameters
    ----------
    y_true : np.ndarray
        Gerçek etiketler.
    y_pred : np.ndarray
        Tahmin edilen etiketler.
    model_name : str
        Model adı.
    task : str
        'binary' veya 'multiclass'.
    """
    y_true = y_true.flatten()
    y_pred = y_pred.flatten()
    avg = 'binary' if task == 'binary' else 'macro'
    
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average=avg, zero_division=0)
    rec = recall_score(y_true, y_pred, average=avg, zero_division=0)
    f1 = f1_score(y_true, y_pred, average=avg, zero_division=0)
    
    print(f"\n{'=' * 50}")
    print(f"  {model_name} - Değerlendirme Sonuçları")
    print(f"{'=' * 50}")
    print(f"  Accuracy  : {acc:.4f}  ({acc*100:.2f}%)")
    print(f"  Precision : {prec:.4f}")
    print(f"  Recall    : {rec:.4f}")
    print(f"  F1 Score  : {f1:.4f}")
    print(f"{'=' * 50}")
    
    print("\n📊 Classification Report:")
    print(classification_report(y_true, y_pred, zero_division=0))
    
    return {'accuracy': acc, 'precision': prec, 'recall': rec, 'f1': f1}


def plot_confusion_matrix(y_true, y_pred, model_name="Model",
                          class_names=None, save_path=None):
    """
    Confusion matrix heatmap çizer.
    
    Parameters
    ----------
    y_true, y_pred : np.ndarray
    model_name : str
    class_names : list, optional
    save_path : str, optional
    """
    y_true = y_true.flatten()
    y_pred = y_pred.flatten()
    
    cm = confusion_matrix(y_true, y_pred)
    
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                xticklabels=class_names, yticklabels=class_names,
                linewidths=0.5, linecolor='gray')
    ax.set_title(f'{model_name}\nKarmaşıklık Matrisi (Confusion Matrix)',
                 fontsize=13, fontweight='bold')
    ax.set_xlabel('Tahmin Edilen', fontsize=11)
    ax.set_ylabel('Gerçek', fontsize=11)
    
    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()


def plot_cost_curves(train_costs, val_costs=None, model_name="Model", save_path=None):
    """
    Eğitim ve validasyon maliyet eğrilerini çizer.
    Overfitting/Underfitting analizi için kullanılır.
    """
    fig, ax = plt.subplots(figsize=(10, 5))
    
    ax.plot(train_costs, label='Train Cost', color='#2196F3', linewidth=2)
    if val_costs:
        ax.plot(val_costs, label='Validation Cost', color='#FF5722',
                linewidth=2, linestyle='--')
    
    ax.set_title(f'{model_name}\nMaliyet Fonksiyonu Değişimi (Cost vs Epoch)',
                 fontsize=13, fontweight='bold')
    ax.set_xlabel('Epoch (Adım)', fontsize=11)
    ax.set_ylabel('Maliyet (Cost)', fontsize=11)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()


def plot_accuracy_curves(train_accs, val_accs=None, model_name="Model", save_path=None):
    """Eğitim ve validasyon doğruluk eğrilerini çizer."""
    fig, ax = plt.subplots(figsize=(10, 5))
    
    ax.plot(train_accs, label='Train Accuracy', color='#4CAF50', linewidth=2)
    if val_accs:
        ax.plot(val_accs, label='Validation Accuracy', color='#FF9800',
                linewidth=2, linestyle='--')
    
    ax.set_title(f'{model_name}\nDoğruluk Değişimi (Accuracy vs Epoch)',
                 fontsize=13, fontweight='bold')
    ax.set_xlabel('Epoch (Adım)', fontsize=11)
    ax.set_ylabel('Doğruluk (Accuracy)', fontsize=11)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 1.05])
    
    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()


def compare_models(results_dict, save_path=None):
    """
    Birden fazla modelin metriklerini karşılaştırır.
    
    Parameters
    ----------
    results_dict : dict
        {model_name: {'accuracy': ..., 'precision': ..., 'recall': ..., 'f1': ...}}
    """
    import pandas as pd
    
    df = pd.DataFrame(results_dict).T
    df.index.name = 'Model'
    
    # Tabloyu yazdır
    print("\n" + "=" * 70)
    print("  MODEL KARŞILAŞTIRMA TABLOSU")
    print("=" * 70)
    print(df.round(4).to_string())
    print("=" * 70)
    
    # Görselleştirme
    fig, ax = plt.subplots(figsize=(12, 6))
    
    x = np.arange(len(df))
    width = 0.2
    metrics = ['accuracy', 'precision', 'recall', 'f1']
    colors = ['#2196F3', '#4CAF50', '#FF9800', '#E91E63']
    
    for i, (metric, color) in enumerate(zip(metrics, colors)):
        if metric in df.columns:
            bars = ax.bar(x + i * width, df[metric], width,
                         label=metric.capitalize(), color=color, edgecolor='black', alpha=0.85)
            # Değer etiketleri
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.005,
                       f'{height:.3f}', ha='center', va='bottom', fontsize=7)
    
    ax.set_title('Model Karşılaştırması', fontsize=14, fontweight='bold')
    ax.set_xlabel('Model', fontsize=11)
    ax.set_ylabel('Skor', fontsize=11)
    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels(df.index, rotation=30, ha='right')
    ax.legend(fontsize=9)
    ax.set_ylim([0, 1.15])
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
    
    return df
