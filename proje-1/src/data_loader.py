"""
Veri yükleme, analiz ve ön işleme modülü.
BankNote Authentication ve Wine Quality veri setleri için DataLoader sınıfı.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler


class DataLoader:
    """
    Veri yükleme, keşifsel analiz ve ön işleme sınıfı.
    
    Parameters
    ----------
    random_state : int
        Tekrarlanabilirlik için rastgele tohum değeri.
    """
    
    def __init__(self, random_state=42):
        self._random_state = random_state
        self._scaler = None
        self.df = None
        self.X_train = None
        self.X_val = None
        self.X_test = None
        self.y_train = None
        self.y_val = None
        self.y_test = None
    
    # ------------------------------------------------------------------ #
    #                        PUBLIC METHODS                                #
    # ------------------------------------------------------------------ #
    
    def load_banknote(self, filepath):
        """BankNote Authentication veri setini yükler."""
        self.df = pd.read_csv(filepath)
        self.dataset_name = "BankNote Authentication"
        self.task = "binary"
        print(f"[INFO] {self.dataset_name} veri seti yüklendi: {self.df.shape}")
        return self.df
    
    def load_wine_quality(self, filepath):
        """
        Wine Quality veri setini yükler (çoklu sınıflandırma).
        
        Parameters
        ----------
        filepath : str
            wine_quality.csv dosyasının yolu.
        
        Returns
        -------
        pd.DataFrame
        """
        self.df = pd.read_csv(filepath)
        self.dataset_name = "Wine Quality"
        self.task = "multiclass"
        
        # Kalite etiketlerini 0'dan başlayacak şekilde dönüştür
        min_quality = self.df['quality'].min()
        self.df['quality'] = self.df['quality'] - min_quality
        self.n_classes = self.df['quality'].nunique()
        self.class_names = [f"Kalite {i + min_quality}" for i in range(self.n_classes)]
        
        print(f"[INFO] {self.dataset_name} veri seti yüklendi: {self.df.shape}")
        print(f"[INFO] Sınıf sayısı: {self.n_classes}, Etiketler: 0..{self.n_classes-1} (Orijinal: {min_quality}..{min_quality + self.n_classes - 1})")
        return self.df
    
    def explore(self):
        """Keşifsel veri analizi (EDA) yapar ve görselleştirir."""
        if self.df is None:
            raise ValueError("Önce veri seti yüklenmelidir!")
        
        print("=" * 60)
        print(f"  {self.dataset_name} - Keşifsel Veri Analizi (EDA)")
        print("=" * 60)
        
        # Temel bilgiler
        print("\n📊 Veri Seti Bilgileri:")
        print(f"   Boyut: {self.df.shape[0]} satır × {self.df.shape[1]} sütun")
        print(f"   Özellikler: {list(self.df.columns[:-1])}")
        print(f"   Hedef: {self.df.columns[-1]}")
        
        print("\n📋 Veri Tipleri:")
        print(self.df.dtypes.to_string())
        
        print("\n🔍 Eksik Değer Kontrolü:")
        null_counts = self.df.isnull().sum()
        if null_counts.sum() == 0:
            print("   ✅ Eksik değer bulunmamaktadır.")
        else:
            print(null_counts[null_counts > 0].to_string())
        
        print("\n📈 İstatistiksel Özet:")
        print(self.df.describe().round(4).to_string())
        
        # Sınıf dağılımı
        print("\n🏷️ Sınıf Dağılımı:")
        class_col = self.df.columns[-1]
        class_counts = self.df[class_col].value_counts().sort_index()
        for cls, cnt in class_counts.items():
            pct = cnt / len(self.df) * 100
            print(f"   Sınıf {cls}: {cnt} örnek ({pct:.1f}%)")
        
        return self.df.describe()
    
    def plot_eda(self, save_dir=None):
        """EDA görselleştirmelerini oluşturur."""
        if self.df is None:
            raise ValueError("Önce veri seti yüklenmelidir!")
        
        features = self.df.columns[:-1]
        class_col = self.df.columns[-1]
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle(f'{self.dataset_name} - Keşifsel Veri Analizi', fontsize=16, fontweight='bold')
        
        # 1. Sınıf dağılımı
        ax = axes[0, 0]
        class_counts = self.df[class_col].value_counts().sort_index()
        colors = sns.color_palette("Set2", len(class_counts))
        bars = ax.bar(class_counts.index.astype(str), class_counts.values, color=colors, edgecolor='black')
        ax.set_title('Sınıf Dağılımı', fontweight='bold')
        ax.set_xlabel('Sınıf')
        ax.set_ylabel('Sayı')
        for bar, count in zip(bars, class_counts.values):
            ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 5,
                    str(count), ha='center', va='bottom', fontweight='bold')
        
        # 2. Korelasyon matrisi
        ax = axes[0, 1]
        corr = self.df[features].corr()
        sns.heatmap(corr, annot=True, cmap='coolwarm', center=0, fmt='.2f',
                    ax=ax, square=True, linewidths=0.5)
        ax.set_title('Korelasyon Matrisi', fontweight='bold')
        
        # 3. Özellik dağılımları (histogram)
        ax = axes[1, 0]
        for i, feat in enumerate(features):
            ax.hist(self.df[feat], bins=30, alpha=0.5, label=feat, edgecolor='black')
        ax.set_title('Özellik Dağılımları', fontweight='bold')
        ax.set_xlabel('Değer')
        ax.set_ylabel('Frekans')
        ax.legend(fontsize=8)
        
        # 4. Box plot
        ax = axes[1, 1]
        self.df[features].boxplot(ax=ax)
        ax.set_title('Box Plot - Özellikler', fontweight='bold')
        ax.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            path = f"{save_dir}/{self.dataset_name.replace(' ', '_')}_eda.png"
            plt.savefig(path, dpi=150, bbox_inches='tight')
            print(f"[INFO] EDA grafiği kaydedildi: {path}")
        
        plt.show()
    
    def split_data(self, test_size=0.15, val_size=0.15, scaling='standard'):
        """
        Veriyi train/validation/test olarak böler ve ölçekler.
        
        Parameters
        ----------
        test_size : float
            Test seti oranı.
        val_size : float
            Validasyon seti oranı.
        scaling : str or None
            'standard', 'minmax' veya None.
        
        Returns
        -------
        tuple : (X_train, X_val, X_test, y_train, y_val, y_test)
        """
        if self.df is None:
            raise ValueError("Önce veri seti yüklenmelidir!")
        
        X = self.df.iloc[:, :-1].values
        y = self.df.iloc[:, -1].values
        
        # Veriyi karıştır
        np.random.seed(self._random_state)
        indices = np.arange(len(X))
        np.random.shuffle(indices)
        X, y = X[indices], y[indices]
        
        # İlk bölme: train+val vs test
        X_temp, self.X_test, y_temp, self.y_test = train_test_split(
            X, y, test_size=test_size, random_state=self._random_state, stratify=y
        )
        
        # İkinci bölme: train vs val
        val_ratio = val_size / (1 - test_size)
        self.X_train, self.X_val, self.y_train, self.y_val = train_test_split(
            X_temp, y_temp, test_size=val_ratio, random_state=self._random_state, stratify=y_temp
        )
        
        # Ölçekleme
        if scaling == 'standard':
            self._scaler = StandardScaler()
            self.X_train = self._scaler.fit_transform(self.X_train)
            self.X_val = self._scaler.transform(self.X_val)
            self.X_test = self._scaler.transform(self.X_test)
            print(f"[INFO] StandardScaler uygulandı.")
        elif scaling == 'minmax':
            self._scaler = MinMaxScaler()
            self.X_train = self._scaler.fit_transform(self.X_train)
            self.X_val = self._scaler.transform(self.X_val)
            self.X_test = self._scaler.transform(self.X_test)
            print(f"[INFO] MinMaxScaler uygulandı.")
        elif scaling is None:
            print(f"[INFO] Ölçekleme uygulanmadı.")
        
        # Etiket şekillendirme
        if self.task == "binary":
            self.y_train = self.y_train.reshape(-1, 1)
            self.y_val = self.y_val.reshape(-1, 1)
            self.y_test = self.y_test.reshape(-1, 1)
        
        print(f"[INFO] Veri bölme tamamlandı:")
        print(f"   Train : {self.X_train.shape[0]} örnek")
        print(f"   Val   : {self.X_val.shape[0]} örnek")
        print(f"   Test  : {self.X_test.shape[0]} örnek")
        
        self._print_class_distribution()
        
        return (self.X_train, self.X_val, self.X_test,
                self.y_train, self.y_val, self.y_test)
    
    # ------------------------------------------------------------------ #
    #                        PRIVATE METHODS                               #
    # ------------------------------------------------------------------ #
    
    def _print_class_distribution(self):
        """Train/Val/Test sınıf dağılımını yazdırır."""
        for name, y in [("Train", self.y_train), ("Val", self.y_val), ("Test", self.y_test)]:
            y_flat = y.flatten().astype(int)
            unique, counts = np.unique(y_flat, return_counts=True)
            dist = dict(zip(unique, counts))
            print(f"   {name} sınıf dağılımı: {dist}")
