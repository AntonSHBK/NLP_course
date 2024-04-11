import torch.nn as nn
import torch.nn.functional as F

class RegressionModel(nn.Module):
    def __init__(self, 
                 input_size, 
                 output_size,
                 layer_sizes=[64], 
                 activation_fn=F.relu):
        super(RegressionModel, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.layer_sizes = layer_sizes
        self.activation_fn = activation_fn
        
        self.layers = nn.ModuleList()
        
        # Создаем слои в соответствии с layer_sizes
        prev_size = input_size
        for size in layer_sizes:
            self.layers.append(nn.Linear(prev_size, size))
            prev_size = size
        
        # Добавляем выходной слой
        self.layers.append(nn.Linear(prev_size, output_size))
        self.activation_fn = activation_fn

    def forward(self, x):
        for layer in self.layers[:-1]:
            x = self.activation_fn(layer(x))
        # Для выходного слоя не применяем функцию активации
        x = self.layers[-1](x)
        return x
