import random
from pathlib import Path

import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from graphviz import Digraph


IMAGES_PATH = Path('../docs/imgs/')
DATA_PATH = Path('../data/project/')

IMAGES_PATH.mkdir(parents=True, exist_ok=True)
DATA_PATH.mkdir(parents=True, exist_ok=True)

def save_plt_img(save_path, fig_id, tight_layout=True, fig_extension="png", resolution=300):
    path = save_path / f"{fig_id}.{fig_extension}"
    if tight_layout:
        plt.tight_layout()
    plt.savefig(path, format=fig_extension, dpi=resolution) 
    
def seed_all(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    random.seed(seed)

def load_data(file_path: Path) -> pd.DataFrame:
    data = pd.read_excel(file_path)    
    return data

def plot_hist(dataframe: pd.DataFrame, column: str):
    category_counts = dataframe.groupby(column).size().sort_values(ascending=False)

    total_messages = category_counts.sum()
    percentages = (category_counts / total_messages * 100).round(2)

    plt.figure(figsize=(12, 8))
    bars = plt.bar(category_counts.index, category_counts.values, color='skyblue')

    plt.title('Распределение сообщений по '+column, fontsize=15, fontweight='bold')
    plt.xlabel(column, fontsize=12)
    plt.ylabel('Количество сообщений', fontsize=12)
    plt.xticks(rotation=45, ha="right")

    for bar, percentage in zip(bars, percentages):
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval, f'{int(yval)}\n({percentage}%)', ha='center', va='bottom', fontsize=7)

    plt.grid(axis='y', linestyle='--', linewidth=0.5)
    plt.tight_layout()
    save_plt_img(save_path=IMAGES_PATH, fig_id=column+'_hist')
    plt.show()
    
def plot_graph():
    # Создаем объект Digraph для генерации диаграммы
    dot = Digraph(comment='Custom GPT-2 Model Architecture')

    # Добавляем узлы, соответствующие компонентам модели
    dot.node('Input', 'Input IDs')
    dot.node('Emb', 'Token Embeddings (wte)')
    dot.node('TypeEmb', 'Type Embeddings')
    dot.node('Combine', 'Combined Embeddings')
    dot.node('Lin', 'Linear Layer (combined_linear)')
    dot.node('GPT2', 'GPT-2 Processing')
    dot.node('Head', 'Output Head (lm_head)')
    dot.node('Logits', 'Logits')

    # Создаем связи между узлами для отображения потока данных
    dot.edge('Input', 'Emb', label='token ids')
    dot.edge('Input', 'TypeEmb', label='type ids')
    dot.edge('Emb', 'Combine', label='embeddings')
    dot.edge('TypeEmb', 'Combine', label='type embeddings')
    dot.edge('Combine', 'Lin', label='combined embeddings')
    dot.edge('Lin', 'GPT2', label='input to GPT-2')
    dot.edge('GPT2', 'Head', label='last hidden state')
    dot.edge('Head', 'Logits', label='output logits')

    # Визуализируем график
    dot.render(IMAGES_PATH/'CustomGPT2ModelArchitecture', format='png', view=False)
