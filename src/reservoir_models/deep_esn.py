# src/reservoir_models/deep_esn.py
import numpy as np
from .esn import IndustrialESN

class DeepESN:
    """Deep Echo State Network با چندین لایه مخزن"""
    
    def __init__(self, n_layers=3, layer_sizes=[100, 200, 100], 
                 inter_scaling=0.5, **esn_kwargs):
        
        self.n_layers = n_layers
        self.layers = []
        
        for i in range(n_layers):
            # هر لایه می‌تواند پارامترهای متفاوتی داشته باشد
            n_inputs = esn_kwargs.get('n_inputs', 1) if i == 0 else layer_sizes[i-1]
            n_reservoir = layer_sizes[i]
            
            layer = IndustrialESN(
                n_inputs=n_inputs,
                n_outputs=layer_sizes[i],
                n_reservoir=n_reservoir,
                spectral_radius=esn_kwargs.get('spectral_radius', 0.9) * (inter_scaling ** i),
                **{k: v for k, v in esn_kwargs.items() if k != 'n_inputs'}
            )
            self.layers.append(layer)
        
        # Readout نهایی
        self.final_readout = Ridge(
            alpha=esn_kwargs.get('regularization', 1e-6)
        )
    
    def fit(self, X, y):
        """آموزش عمیق"""
        # انتشار forward در لایه‌ها
        layer_outputs = []
        current_input = X
        
        for i, layer in enumerate(self.layers):
            # آموزش یا forward pass
            if i == 0:
                layer.fit(current_input, current_input)  # Self-supervised برای لایه اول
            else:
                layer.fit(current_input, current_input)
            
            # جمع‌آوری خروجی لایه
            layer_pred, _ = layer.predict(current_input, return_states=True)
            layer_outputs.append(layer_pred)
            current_input = layer_pred
        
        # ترکیب خروجی همه لایه‌ها برای readout نهایی
        all_features = np.hstack(layer_outputs)
        self.final_readout.fit(all_features, y)
        
        return self
