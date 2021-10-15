import os

import numpy as np
from transformers import AutoModelForSequenceClassification, BertForSequenceClassification, BertTokenizer, Trainer, TrainingArguments


class QualityModelLabeler:
    
    def __init__(self, fit_me=False, path=None, bert_type='bert-base-uncased'):
        
        self.path = path
        self.bert_type = bert_type
        
        self.tokenizer = BertTokenizer.from_pretrained(bert_type)
        
        if not fit_me:
            self.model = AutoModelForSequenceClassification.from_pretrained(path)
            self.trainer = Trainer(model=self.model)
        else:
            
            self.model = BertForSequenceClassification.from_pretrained(bert_type)
            self.trainer = None # To be initialized
    
    def fit(self, raw_texts, labels):
        
        all_tokenized = self.tokenizer(raw_texts, return_attention_mask=True, is_split_into_words=False, padding=True, truncation=True, return_tensors='pt')
        all_tokenized['label'] = labels
        
        # Rearrange data for training
        all_data = []
        for i in range(len(labels)):
            d = {k: all_tokenized[k][i] for k in all_tokenized.keys()}
            all_data.append(d)
        
        self.model = BertForSequenceClassification.from_pretrained(self.bert_type, num_labels=1)
        args = TrainingArguments(num_train_epochs=3, 
                                 learning_rate=2e-5,   
                                 output_dir=os.path.expanduser('~/quality_temp'),
                                 per_device_train_batch_size=16,
                                 logging_steps=200)

        self.trainer = Trainer(
                        self.model,
                        args=args,
                        train_dataset=all_data,
                    )
        self.trainer.train()
        
        self.trainer.save_model(self.path)
        
    def label_sent_stats(self, text):
        
        if self.trainer is None:
            raise ValueError("No Trained Model Here!!")
        
        sents = [s.strip() for s in text.split('.') if len(s) > 25]
    
        if len(sents) == 0:
            return None
        
        tokenized_test = self.tokenizer(sents, return_attention_mask=True, is_split_into_words=False, padding=True, truncation=True, return_tensors='pt')
    
        cur_vectors = [{k: v[idx] for k, v in tokenized_test.items()} for idx in range(len(sents))]

        row = self.trainer.predict(cur_vectors).predictions.reshape(-1)
        
        stats = {'max': row.max(), 'mean': row.mean(), 'min': row.min(), 'range': row.max() - row.min(),
                 'p50': np.percentile(row, 50), 'p25': np.percentile(row, 25), 'p75': np.percentile(row, 75)}
        
        return stats
    
    
if __name__ == '__main__':
    import pandas as pd
    
    model_path=os.path.expanduser('~/final_paper_data_v2/models/final_ibm_quality/')
    
    data = pd.read_csv(os.path.expanduser('~/final_paper_data_v2/arg_quality_rank_30k.csv'))
    print("Now training on quality data")
    print(data.head())
    
    texts = data.argument.tolist()
    labels = data['MACE-P'].tolist()
    print(texts[:2])
    print(labels[:2])
    
    qmodel = QualityModelLabeler(fit_me=True, path=model_path)
    
    qmodel.fit(texts, labels)
    