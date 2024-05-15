from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from nltk.translate.bleu_score import corpus_bleu
from rouge import Rouge
from nltk.translate.meteor_score import meteor_score

from utils import save_plt_img


class Evaluator:
    def __init__(self, tokenizer, weights={'bleu': 0.34, 'rouge': 0.33, 'meteor': 0.33}):
        self.tokenizer = tokenizer
        self.weights = weights
        self.rouge = Rouge()
        
        self.metrics = []

    def evaluate(self, references: list, hypotheses: list):
        tokenized_references = [[ref.split()] for ref in references]
        tokenized_hypotheses = [hyp.split() for hyp in hypotheses]
        
        # Вычисление метрик BLEU, ROUGE, METEOR
        bleu_score = corpus_bleu(tokenized_references, tokenized_hypotheses)
        
        rouge_score = self.rouge.get_scores(hypotheses, references, avg=True)['rouge-l']['f']
        
        list_meteor_score = [meteor_score(refs, hyp) for refs, hyp in zip(tokenized_references, tokenized_hypotheses)]
        avg_meteor_score = np.mean(list_meteor_score)

        # Словарь с результатами
        results = {
            'overall': self.weights['bleu'] * bleu_score +
                       self.weights['rouge'] * rouge_score +
                       self.weights['meteor'] * avg_meteor_score,
            'bleu': bleu_score,
            'rouge': rouge_score,
            'meteor': avg_meteor_score
        }
        self.metrics.append(results)
        return results
    
    def plot_metrics(self, imgs_path=Path(''), name='special_metrics'):
        epochs_range = range(1, len(self.metrics) + 1)
        fig = plt.figure(figsize=(15, 6))  # Устанавливаем размер фигуры
        
        ax12 = plt.subplot(1, 2, 1)
        ax12.plot(epochs_range, [m['overall'] for m in self.metrics], label='Overall Score', color='tab:green')
        ax12.set_title('Overall Evaluation Score')
        ax12.set_xlabel('Epochs')
        ax12.set_ylabel('Score')
        ax12.grid(True, which='both', linestyle='--', linewidth=0.5)
        ax12.legend(loc='upper right')

        # Отдельные графики для каждой метрики
        ax21 = plt.subplot(1, 2, 2)
        ax21.plot(epochs_range, [m['bleu'] for m in self.metrics], label='BLEU Score', color='tab:red')
        ax21.plot(epochs_range, [m['rouge'] for m in self.metrics], label='ROUGE Score', color='tab:pink')
        ax21.plot(epochs_range, [m['meteor'] for m in self.metrics], label='METEOR Score', color='tab:brown')
        ax21.set_title('Individual Metrics')
        ax21.set_xlabel('Epochs')
        ax21.set_ylabel('Score')
        ax21.grid(True, which='both', linestyle='--', linewidth=0.5)
        ax21.legend(loc='upper right')      

        fig.tight_layout()  # Убедимся, что макет не нарушен
        save_plt_img(imgs_path, name)
        plt.show()
