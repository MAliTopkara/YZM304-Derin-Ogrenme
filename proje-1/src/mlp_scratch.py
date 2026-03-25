"""
Sıfırdan (from scratch) çok katmanlı algılayıcı (MLP) sınıfı.
Binary ve Multi-class sınıflandırma desteği.
Numpy tabanlı, OOP yapısında.
"""

import numpy as np
from src.utils import sigmoid, sigmoid_derivative, tanh, tanh_derivative, softmax, one_hot_encode


class MLPScratch:
    """
    Sıfırdan yazılmış çok katmanlı algılayıcı (Multi-Layer Perceptron).
    
    Parameters
    ----------
    layer_dims : list
        Katman boyutları listesi. Örn: [4, 5, 1] veya [4, 8, 4, 3]
    learning_rate : float
        Öğrenme oranı (SGD için).
    n_steps : int
        Eğitim iterasyon sayısı (epoch).
    random_state : int
        Ağırlık başlatma için rastgele tohum.
    lambda_reg : float
        L2 regülarizasyon katsayısı. 0 ise regülarizasyon yok.
    task : str
        'binary' veya 'multiclass'.
    activation_hidden : str
        Gizli katman aktivasyon fonksiyonu: 'tanh' veya 'relu'.
    """
    
    def __init__(self, layer_dims, learning_rate=0.01, n_steps=1000,
                 random_state=42, lambda_reg=0.0, task='binary',
                 activation_hidden='tanh'):
        self.layer_dims = layer_dims
        self.learning_rate = learning_rate
        self.n_steps = n_steps
        self.random_state = random_state
        self.lambda_reg = lambda_reg
        self.task = task
        self.activation_hidden = activation_hidden
        
        self.parameters = {}
        self.cost_history_train = []
        self.cost_history_val = []
        self.accuracy_history_train = []
        self.accuracy_history_val = []
        self._n_layers = len(layer_dims) - 1  # Ağırlık katmanı sayısı
    
    # ================================================================== #
    #                        PUBLIC METHODS                                #
    # ================================================================== #
    
    def fit(self, X_train, y_train, X_val=None, y_val=None, print_cost=True, print_every=100):
        """
        Modeli eğitir.
        
        Parameters
        ----------
        X_train : np.ndarray, shape (m_train, n_features)
            Eğitim verileri.
        y_train : np.ndarray
            Eğitim etiketleri.
        X_val : np.ndarray, optional
            Validasyon verileri.
        y_val : np.ndarray, optional
            Validasyon etiketleri.
        print_cost : bool
            Maliyet değerlerini yazdır.
        print_every : int
            Her kaç adımda bir yazdırılacağı.
        """
        # Parametreleri başlat
        self._initialize_parameters()
        
        self.cost_history_train = []
        self.cost_history_val = []
        self.accuracy_history_train = []
        self.accuracy_history_val = []
        
        for i in range(self.n_steps):
            # İleri yayılım
            AL, caches = self._forward_propagation(X_train)
            
            # Maliyet hesapla
            cost_train = self._compute_cost(AL, y_train)
            self.cost_history_train.append(cost_train)
            
            # Eğitim doğruluğu
            acc_train = self._compute_accuracy(X_train, y_train)
            self.accuracy_history_train.append(acc_train)
            
            # Validasyon seti varsa
            if X_val is not None and y_val is not None:
                AL_val, _ = self._forward_propagation(X_val)
                cost_val = self._compute_cost(AL_val, y_val)
                self.cost_history_val.append(cost_val)
                acc_val = self._compute_accuracy(X_val, y_val)
                self.accuracy_history_val.append(acc_val)
            
            # Geri yayılım
            grads = self._backward_propagation(AL, y_train, caches)
            
            # Parametre güncelleme (SGD)
            self._update_parameters(grads)
            
            # Yazdır
            if print_cost and i % print_every == 0:
                msg = f"Step {i:5d} | Train Cost: {cost_train:.6f} | Train Acc: {acc_train:.4f}"
                if X_val is not None:
                    msg += f" | Val Cost: {cost_val:.6f} | Val Acc: {acc_val:.4f}"
                print(msg)
        
        # Son adım bilgisi
        if print_cost:
            print("-" * 70)
            final_msg = f"Step {self.n_steps:5d} | Train Cost: {self.cost_history_train[-1]:.6f} | Train Acc: {self.accuracy_history_train[-1]:.4f}"
            if X_val is not None:
                final_msg += f" | Val Cost: {self.cost_history_val[-1]:.6f} | Val Acc: {self.accuracy_history_val[-1]:.4f}"
            print(final_msg)
        
        return self
    
    def predict(self, X):
        """
        Tahmin yapar.
        
        Parameters
        ----------
        X : np.ndarray, shape (m, n_features)
        
        Returns
        -------
        np.ndarray : Tahmin edilen sınıf etiketleri.
        """
        AL, _ = self._forward_propagation(X)
        
        if self.task == 'binary':
            predictions = (AL > 0.5).astype(int)
            return predictions.flatten()
        else:
            # Multi-class: en yüksek olasılıklı sınıf
            predictions = np.argmax(AL, axis=0)
            return predictions
    
    def predict_proba(self, X):
        """Olasılık çıktısı döndürür."""
        AL, _ = self._forward_propagation(X)
        return AL
    
    def evaluate(self, X, y):
        """
        Modeli değerlendirir.
        
        Returns
        -------
        dict : accuracy, precision, recall, f1 değerleri.
        """
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
        
        y_pred = self.predict(X)
        y_true = y.flatten()
        
        avg = 'binary' if self.task == 'binary' else 'macro'
        
        results = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, average=avg, zero_division=0),
            'recall': recall_score(y_true, y_pred, average=avg, zero_division=0),
            'f1': f1_score(y_true, y_pred, average=avg, zero_division=0)
        }
        return results
    
    def get_params(self):
        """Model parametrelerini ve hiperparametreleri döndürür."""
        return {
            'layer_dims': self.layer_dims,
            'learning_rate': self.learning_rate,
            'n_steps': self.n_steps,
            'random_state': self.random_state,
            'lambda_reg': self.lambda_reg,
            'task': self.task,
            'activation_hidden': self.activation_hidden,
            'parameters': self.parameters
        }
    
    # ================================================================== #
    #                        PRIVATE METHODS                               #
    # ================================================================== #
    
    def _initialize_parameters(self):
        """
        Ağırlıkları ve bias'ları başlatır.
        Xavier initialization (tanh için) veya He initialization (relu için).
        """
        np.random.seed(self.random_state)
        self.parameters = {}
        
        for l in range(1, self._n_layers + 1):
            if self.activation_hidden == 'relu':
                scale = np.sqrt(2.0 / self.layer_dims[l-1])  # He init
            else:
                scale = np.sqrt(1.0 / self.layer_dims[l-1])  # Xavier init
            self.parameters[f'W{l}'] = np.random.randn(
                self.layer_dims[l], self.layer_dims[l-1]
            ) * scale
            self.parameters[f'b{l}'] = np.zeros((self.layer_dims[l], 1))
    
    def _forward_propagation(self, X):
        """
        N-katmanlı ileri yayılım.
        
        Gizli katmanlar: tanh (veya relu)
        Çıkış katmanı: sigmoid (binary) veya softmax (multiclass)
        
        Parameters
        ----------
        X : np.ndarray, shape (m, n_features)
        
        Returns
        -------
        AL : np.ndarray
            Çıkış katmanı aktivasyonu.
        caches : list
            Her katman için (Z, A_prev) cache'i.
        """
        A = X.T  # shape: (n_features, m)
        caches = []
        
        # Gizli katmanlar (1 ... L-1)
        for l in range(1, self._n_layers):
            A_prev = A
            W = self.parameters[f'W{l}']
            b = self.parameters[f'b{l}']
            Z = np.dot(W, A_prev) + b
            
            if self.activation_hidden == 'tanh':
                A = tanh(Z)
            elif self.activation_hidden == 'relu':
                from src.utils import relu
                A = relu(Z)
            
            caches.append({'Z': Z, 'A_prev': A_prev, 'A': A, 'W': W})
        
        # Çıkış katmanı (L)
        A_prev = A
        W = self.parameters[f'W{self._n_layers}']
        b = self.parameters[f'b{self._n_layers}']
        Z = np.dot(W, A_prev) + b
        
        if self.task == 'binary':
            AL = sigmoid(Z)
        else:
            AL = softmax(Z)
        
        caches.append({'Z': Z, 'A_prev': A_prev, 'A': AL, 'W': W})
        
        return AL, caches
    
    def _compute_cost(self, AL, Y):
        """
        Kayıp fonksiyonunu hesaplar.
        
        Binary: Binary Cross Entropy
        Multiclass: Categorical Cross Entropy
        L2 regülarizasyon opsiyonel.
        """
        if self.task == 'binary':
            Y_T = Y.flatten().reshape(1, -1)  # shape: (1, m)
            m = Y_T.shape[1]
            
            # Numerik kararlılık
            AL_clipped = np.clip(AL, 1e-8, 1 - 1e-8)
            
            cost = -(1.0 / m) * np.sum(
                Y_T * np.log(AL_clipped) + (1 - Y_T) * np.log(1 - AL_clipped)
            )
        else:
            # Categorical Cross Entropy
            m = AL.shape[1]
            n_classes = self.layer_dims[-1]
            Y_onehot = one_hot_encode(Y, n_classes)
            
            AL_clipped = np.clip(AL, 1e-8, 1 - 1e-8)
            cost = -(1.0 / m) * np.sum(Y_onehot * np.log(AL_clipped))
        
        # L2 Regülarizasyon
        if self.lambda_reg > 0:
            m = AL.shape[1]
            l2_cost = 0
            for l in range(1, self._n_layers + 1):
                l2_cost += np.sum(np.square(self.parameters[f'W{l}']))
            cost += (self.lambda_reg / (2 * m)) * l2_cost
        
        return float(np.squeeze(cost))
    
    def _backward_propagation(self, AL, Y, caches):
        """
        N-katmanlı geri yayılım.
        
        Returns
        -------
        grads : dict
            Her katman için dW, db gradyanları.
        """
        grads = {}
        m = AL.shape[1]
        
        if self.task == 'binary':
            Y_T = Y.flatten().reshape(1, -1)  # shape: (1, m)
        else:
            n_classes = self.layer_dims[-1]
            Y_T = one_hot_encode(Y, n_classes)  # shape: (n_classes, m)
        
        # Çıkış katmanı gradyanı
        # Sigmoid/Softmax + Cross Entropy birlikte: dZ_L = AL - Y
        dZ = AL - Y_T
        
        cache = caches[self._n_layers - 1]
        A_prev = cache['A_prev']
        
        grads[f'dW{self._n_layers}'] = (1.0 / m) * np.dot(dZ, A_prev.T)
        grads[f'db{self._n_layers}'] = (1.0 / m) * np.sum(dZ, axis=1, keepdims=True)
        
        # L2 regülarizasyon gradyanı
        if self.lambda_reg > 0:
            grads[f'dW{self._n_layers}'] += (self.lambda_reg / m) * self.parameters[f'W{self._n_layers}']
        
        # Gizli katmanlar (L-1 ... 1)
        dA_prev = np.dot(self.parameters[f'W{self._n_layers}'].T, dZ)
        
        for l in reversed(range(1, self._n_layers)):
            cache = caches[l - 1]
            A_l = cache['A']
            A_prev = cache['A_prev']
            
            if self.activation_hidden == 'tanh':
                dZ = dA_prev * tanh_derivative(A_l)
            elif self.activation_hidden == 'relu':
                from src.utils import relu_derivative
                dZ = dA_prev * relu_derivative(cache['Z'])
            
            grads[f'dW{l}'] = (1.0 / m) * np.dot(dZ, A_prev.T)
            grads[f'db{l}'] = (1.0 / m) * np.sum(dZ, axis=1, keepdims=True)
            
            # L2 regülarizasyon gradyanı
            if self.lambda_reg > 0:
                grads[f'dW{l}'] += (self.lambda_reg / m) * self.parameters[f'W{l}']
            
            if l > 1:
                dA_prev = np.dot(self.parameters[f'W{l}'].T, dZ)
        
        return grads
    
    def _update_parameters(self, grads):
        """SGD ile parametre güncelleme."""
        for l in range(1, self._n_layers + 1):
            self.parameters[f'W{l}'] -= self.learning_rate * grads[f'dW{l}']
            self.parameters[f'b{l}'] -= self.learning_rate * grads[f'db{l}']
    
    def _compute_accuracy(self, X, y):
        """Doğruluk hesaplar."""
        y_pred = self.predict(X)
        y_true = y.flatten()
        return np.mean(y_pred == y_true)
    
    def __repr__(self):
        arch = " → ".join(str(d) for d in self.layer_dims)
        return (f"MLPScratch(arch=[{arch}], lr={self.learning_rate}, "
                f"steps={self.n_steps}, task={self.task}, λ={self.lambda_reg})")
