from pathlib import Path

import pandas as pd
import torch
import torch.nn.functional as F
from torch import Tensor
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader
from transformers import AdamW
from transformers import GPT2Tokenizer
from tqdm.auto import tqdm
from torchviz import make_dot
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


from model import CustomGPT2Model, Config
from evaluator import Evaluator
from utils import save_plt_img
from custom_data import CategoricalLabelEncoder, CustomDataset





class Trainer:
    def __init__(
        self, 
        model: CustomGPT2Model, 
        evaluator: Evaluator, 
        tokenizer,
        learning_rate=1e-3,
        special_eval=False,
        device='cpu',
        config: Config=None
    ):
        self.model = model.to(device)
        self.special_eval = special_eval
        self.device = device
        self.evaluator =  evaluator
        self.tokenizer = tokenizer
        self.config = config
        
        self.train_losses = []
        self.val_losses = []
        
        self.optimizer = AdamW(model.parameters(), lr=learning_rate)
        self.criterion = CrossEntropyLoss()
        
        self.freeze_layers()
        self.print_freezed_layers()

    def train(self, train_loader: DataLoader):
        """
        Обучение модели
        """
        self.model.train()
        train_loss = 0.0
        train_bar = tqdm(train_loader, desc='Training', leave=True)
        for batch in train_bar:
            inputs_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            target_ids = batch['target_ids'].to(self.device)
            type_ids = batch['type_ids'].to(self.device)
            
            logits = self.model(input_ids=inputs_ids, attention_mask=attention_mask, type_ids=type_ids)
            loss = self.criterion(
                logits.view(-1, self.model.gpt2.config.vocab_size),
                target_ids.view(-1)
            )
            
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            train_loss += loss.item()
            train_bar.set_postfix({'train loss': loss.item()})
            
        avg_loss = train_loss / len(train_loader)
        self.train_losses.append(avg_loss)        
        return avg_loss
            

    def eval(self, eval_loader: DataLoader):
        self.model.eval()
        eval_loss = 0
        eval_bar = tqdm(eval_loader, desc=f'Evaluate', leave=True)
        with torch.no_grad():
            for batch in eval_bar:
                inputs_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                target_ids = batch['target_ids'].to(self.device)
                type_ids = batch['type_ids'].to(self.device)
                
                logits = self.model(input_ids=inputs_ids, attention_mask=attention_mask, type_ids=type_ids)

                loss = self.criterion(
                    logits.view(-1, self.model.gpt2.config.vocab_size),
                    target_ids.view(-1)
                )                          
                eval_loss += loss.item()
                eval_bar.set_postfix({'eval loss': loss.item()})
                
            avg_loss = eval_loss / len(eval_loader)
            self.val_losses.append(avg_loss)    
        
        self.calculate_special_eval(eval_loader, self.config.max_length, self.config.temperature)
        
        return avg_loss
            
    def fit(self, epochs, train_loader: DataLoader, eval_loader: DataLoader):
        epoch_bar = tqdm(range(epochs), desc=f'Fit ', leave=True)
        for epoch in epoch_bar:
            train_avg_loss = self.train(train_loader)
            eval_avg_loss = self.eval(eval_loader)
            
            # Обновление прогресс-бара с текущей эпохой и значениями потерь
            epoch_bar.set_postfix({
                'Epoch': epoch,
                'Train Loss': f"{train_avg_loss:.4f}",
                'Eval Loss': f"{eval_avg_loss:.4f}"
            })
        
    def calculate_special_eval(self, eval_loader: DataLoader, max_length, temperature):
        if self.special_eval and self.evaluator:
            self.model.eval()
            eval_bar = tqdm(eval_loader, desc=f'Special evaluate', leave=True)
            hypotheses = []
            references = []
            
            with torch.no_grad():
                for batch in eval_bar:
                    input_ids = batch['input_ids'].to(self.device)
                    attention_mask = batch['attention_mask'].to(self.device)
                    target_ids = batch['target_ids'].to(self.device)
                    type_ids = batch['type_ids'].to(self.device)
                    
                    hypotheses += self._generate_text_sampling(input_ids, attention_mask, type_ids, max_length, temperature)
                    references += self.decode_tensor(target_ids)
            self.evaluator.evaluate(hypotheses=hypotheses, references=references)
        
    def generate_text(self, text, type_id, max_length=25, method='argmax', temperature=0.9, top_k=10, top_p=0.9):
        # Токенизация входного текста
        inputs = self.tokenizer(text, return_tensors="pt")
        input_ids = inputs['input_ids'].to(self.device)
        attention_mask = inputs['attention_mask'].to(self.device)
        type_ids = torch.tensor([type_id]).to(self.device)
        
        if method == 'argmax':
            return self._generate_text_argmax(input_ids, attention_mask, type_ids, max_length)
        elif method == 'sampling':
            return self._generate_text_sampling(input_ids, attention_mask, type_ids, max_length, temperature)
        elif method == 'top_k':
            return self._generate_text_top_k(input_ids, attention_mask, type_ids, max_length, k=top_k)
        elif method == 'top_p':
            return self._generate_text_top_p(input_ids, attention_mask, type_ids, max_length, p=top_p)
            
    def _generate_text_sampling(self, input_ids, attention_mask, type_ids, max_length=25, temperature=0.9):
        self.model.eval() 
        generated_sequence = input_ids   
        
        # Начальная длина входных данных
        start_gen_index = input_ids.size(1) 
        
        with torch.no_grad():
            for _ in range(max_length):
                outputs = self.model(input_ids=generated_sequence, attention_mask=attention_mask, type_ids=type_ids)
                logits = outputs[:, -1, :]                
                # Применяем температуру для управления случайностью выборки
                logits = logits / temperature
                probabilities = F.softmax(logits, dim=-1)
                next_token_id = torch.multinomial(probabilities, num_samples=1)                
                generated_sequence = torch.cat((generated_sequence, next_token_id), dim=-1)
                
                # Обновление attention_mask
                attention_mask = torch.cat([attention_mask, torch.ones((attention_mask.size(0), 1), device=self.device)], dim=1)

        # Возвращаем только сгенерированную часть
        generated_part = generated_sequence[:, start_gen_index:]
        return self.decode_tensor(generated_part)
    
    def _generate_text_argmax(self, input_ids, attention_mask, type_ids, max_length=25):
        self.model.eval()
        generated_sequence = input_ids

        # Начальная длина входных данных
        start_gen_index = input_ids.size(1)

        with torch.no_grad():
            for _ in range(max_length):
                outputs = self.model(input_ids=generated_sequence, attention_mask=attention_mask, type_ids=type_ids)
                                
                logits = outputs[:, -1, :]
                                
                next_token_id = torch.argmax(logits, dim=-1, keepdim=True)
                generated_sequence = torch.cat((generated_sequence, next_token_id), dim=-1)
                
                # Обновление attention_mask
                attention_mask = torch.cat([attention_mask, torch.ones((attention_mask.size(0), 1), device=self.device)], dim=1)
        
        # Возвращаем только сгенерированную часть
        generated_part = generated_sequence[:, start_gen_index:]
        return self.decode_tensor(generated_part)
    
    def _generate_text_top_k(self, input_ids, attention_mask, type_ids, max_length=25, k=10):
        self.model.eval()
        generated_sequence = input_ids

        with torch.no_grad():
            for _ in range(max_length):
                outputs = self.model(input_ids=generated_sequence, attention_mask=attention_mask, type_ids=type_ids)
                logits = outputs[:, -1, :]
                # Применяем Top-k sampling
                top_k = torch.topk(logits, k=k, dim=-1)
                probabilities = torch.softmax(top_k.values, dim=-1)
                next_token_id = torch.multinomial(probabilities, num_samples=1)
                next_token_id = top_k.indices.gather(-1, next_token_id)
                
                generated_sequence = torch.cat((generated_sequence, next_token_id), dim=-1)
                attention_mask = torch.cat([attention_mask, torch.ones((1, 1), device=self.device)], dim=1)

        return self.decode_tensor(generated_sequence[:, input_ids.size(1):])

    def _generate_text_top_p(self, input_ids, attention_mask, type_ids, max_length=25, p=0.9):
        self.model.eval()
        generated_sequence = input_ids

        with torch.no_grad():
            for _ in range(max_length):
                outputs = self.model(input_ids=generated_sequence, attention_mask=attention_mask, type_ids=type_ids)
                logits = outputs[:, -1, :]
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)

                # Удаление токенов с суммой вероятностей выше p
                sorted_indices_to_remove = cumulative_probs > p
                # Оставляем хотя бы один токен, если все превышают p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = False

                indices_to_remove = sorted_indices[sorted_indices_to_remove]
                logits[:, indices_to_remove] = float('-inf')
                probabilities = torch.softmax(logits, dim=-1)
                next_token_id = torch.multinomial(probabilities, num_samples=1)

                generated_sequence = torch.cat((generated_sequence, next_token_id), dim=-1)
                attention_mask = torch.cat([attention_mask, torch.ones((1, 1), device=self.device)], dim=1)

        return self.decode_tensor(generated_sequence[:, input_ids.size(1):])
    
    def decode_tensor(self, sequences: Tensor):
        decoded_texts = []
        for sequence in sequences:
            decoded_text = self.tokenizer.decode(sequence, skip_special_tokens=True)
            decoded_texts.append(decoded_text)
        return decoded_texts

    def plot_main_metrics(self, imgs_path=Path(''), name='train_metrics'):
        epochs_range = range(1, len(self.val_losses) + 1)
        fig = plt.figure(figsize=(15, 6))  # Устанавливаем размер фигуры
        
        ax11 = plt.subplot()
        ax11.plot(epochs_range, self.train_losses, label='Train loss', color='tab:red')
        ax11.plot(epochs_range, self.val_losses, label='Evaluate loss', color='tab:blue')
        ax11.set_title('Losses over Epochs')
        ax11.set_xlabel('Epochs')
        ax11.set_ylabel('Loss')
        ax11.grid(True, which='both', linestyle='--', linewidth=0.5)
        ax11.legend(loc='upper right')     

        fig.tight_layout()  # Убедимся, что макет не нарушен
        save_plt_img(save_path=imgs_path, 
                     fig_id=name)  # extra code
        plt.show()
            
    def freeze_layers(self, num_trainable_blocks=1):        
        total_blocks = len(self.model.gpt2.h)  # h - это список всех блоков трансформера в модели GPT-2

        # Заморозка слоёв в начальных блоках
        for i, block in enumerate(self.model.gpt2.h):
            if i < total_blocks - num_trainable_blocks:
                for param in block.parameters():
                    param.requires_grad = False
            else:
                for param in block.parameters():
                    param.requires_grad = True

        # # Если у вас есть другие специфические слои, которые нужно обучать, размораживаем их
        for param in self.model.combined_linear.parameters():
            param.requires_grad = True
        
        for param in self.model.type_embedding.parameters():
            param.requires_grad = True
    
    def print_freezed_layers(self):
        for name, param in self.model.named_parameters():
            print(f"{name} is trainable: {param.requires_grad}")
    
    def save_model(self, file_path: Path):
        # Словарь для сохранения
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': self.config
        }
        torch.save(checkpoint, file_path)
        print(f"Model saved to {file_path}")
    
    def load_model(self, file_path: Path):
        checkpoint = torch.load(file_path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.model.to(self.device)  # Убедитесь, что модель перенесена на нужное устройство
        print(f"Model loaded from {file_path}")
        
    def render_graaph(self, output, imgs_path=Path(''), name='neural_network_graph'):
        # Визуализация графа
        dot = make_dot(output, params=dict(self.model.named_parameters()))
        dot.attr('node', style='filled', color='lightblue')
        dot.attr('edge', style='dashed', color='grey')                   
        dot.format = 'png'  # Устанавливаем формат файла
        # Можно установить разрешение в dpi (точек на дюйм), например 300 dpi
        dot.attr(dpi='300')
        dot.render(filename=name,
                   directory=imgs_path)  # Сохраняем граф в файл


class TrainingManager:
    def __init__(self, 
        dataframe: pd.DataFrame,
        categorical_encoder: CategoricalLabelEncoder,
        config: Config,
        data_path: Path='',
        imgs_path: Path='' 
    ):
        
        self.uniq_name = config.uniq_name
        self.data_path = data_path 
        self.imgs_path = imgs_path    
        self.dataframe = dataframe
        self.categorical_encoder = categorical_encoder
                
        self.config = config
        
        self.tokenizer = GPT2Tokenizer.from_pretrained(config.model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Разбиение данных на обучающую и тестовую выборки
        train_data, eval_data = train_test_split(
            self.dataframe[['type_problem', 'source', 'target']], test_size=0.1
        )

        # Создание объектов Dataset
        train_dataset = CustomDataset(dataframe=train_data, tokenizer=self.tokenizer, max_length=config.max_length)
        eval_dataset = CustomDataset(dataframe=eval_data, tokenizer=self.tokenizer, max_length=config.max_length)

        self.train_dataloader = DataLoader(
            train_dataset, 
            batch_size=config.batch_size, 
            shuffle=True
        )
        self.eval_dataloader = DataLoader(
            eval_dataset, 
            batch_size=config.batch_size, 
            shuffle=False
        )

        self.model = CustomGPT2Model(
            pretrained_model_name=config.model_name,
            num_message_types=len(self.categorical_encoder.get_classes('type_problem')),
            data_path=self.data_path
        )
        
        evaluator = Evaluator(
            self.tokenizer 
        )
        
        self.trainer = Trainer(
            model=self.model,
            evaluator=evaluator,
            tokenizer=self.tokenizer,
            learning_rate=config.learning_rate,
            device=config.device,
            config=config,
            special_eval=config.special_eval           
        )  
    
    def fit(self):
        """
        Запуск процесса обучения.
        """
        
        self.trainer.fit(self.config.num_epochs, self.train_dataloader, self.eval_dataloader)
        
    def plot_main_metrics(self):
        name = self.uniq_name + '_main_metrics'
        self.trainer.plot_main_metrics(self.imgs_path, name)
        
    def plot_special_metrics(self):
        name = self.uniq_name + '_special_metrrics'
        self.trainer.evaluator.plot_metrics(self.imgs_path, name)
    
    def generate_text(self, text, type_id, max_length=25, method='samplig', **kwargs):
        return self.trainer.generate_text(text, type_id, max_length, method=method, **kwargs)[0]

    def save(self):
        self.trainer.save_model(self.data_path / (self.uniq_name + '.pth'))
 
    def load(self):
        self.trainer.load_model(self.data_path / (self.uniq_name + '.pth'))
