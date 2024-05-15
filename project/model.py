from pathlib import Path

import torch
import torch.nn as nn
from transformers import GPT2Model

class CustomGPT2Model(nn.Module):
    def __init__(
        self, 
        pretrained_model_name,
        num_message_types,
        data_path=Path('')
    ):
        super(CustomGPT2Model, self).__init__()
        # Загрузка предобученной модели GPT-2
        self.gpt2 = GPT2Model.from_pretrained(pretrained_model_name, cache_dir=data_path)
        
        # Слой для встраивания типа сообщения
        self.type_embedding = nn.Embedding(num_message_types, self.gpt2.config.hidden_size)
        
        # Дополнительный линейный слой для объединения встраивания типа сообщения и токенов
        self.combined_linear = nn.Linear(self.gpt2.config.hidden_size, self.gpt2.config.hidden_size)
        # Дополнительный слой для логитов
        self.lm_head = nn.Linear(self.gpt2.config.hidden_size, self.gpt2.config.vocab_size)  # Дополнительный слой для логитов


    def forward(self, input_ids, attention_mask, type_ids):
        # Получение встраиваний токенов
        inputs_embeds = self.gpt2.wte(input_ids)  # wte - word token embeddings
        
        # Получение встраивания для типа сообщения
        type_embeds = self.type_embedding(type_ids).unsqueeze(1)  # Расширяем размерности для сложения
        
        # Сложение встраиваний токенов и типа по всей длине входа
        combined_embeds = inputs_embeds + type_embeds.expand(-1, input_ids.size(1), -1)
        
        # Применяем дополнительный линейный слой
        combined_embeds = self.combined_linear(combined_embeds)
        
        # Передача встраиваний в основную модель GPT-2
        outputs = self.gpt2(inputs_embeds=combined_embeds, attention_mask=attention_mask)
        
        logits = self.lm_head(outputs.last_hidden_state)
        return logits
    

class Config:
    model_name = 'ai-forever/rugpt3small_based_on_gpt2'
    max_length = 64
    batch_size = 128
    test_size=0.1
    learning_rate = 1e-5
    num_epochs = 30
    uniq_name = 'custom_gpt2_model'
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    special_eval = False
    
    temperature=0.7
    top_k=11
    top_p=0.9