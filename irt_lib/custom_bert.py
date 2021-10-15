from transformers import BertPreTrainedModel, BertModel, BertTokenizer
from transformers.modeling_outputs import SequenceClassifierOutput

import torch
from torch import nn

class BertMultiHead(BertPreTrainedModel):
    
    def __init__(self, config, num_heads):
        super().__init__(config)
        
        self.num_labels = config.num_labels
        
        self.num_heads = num_heads
        
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        self.classifier_heads = nn.ModuleList([nn.Linear(config.hidden_size, config.num_labels) for i in range(self.num_heads)])
        
        self.init_weights()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        labels=None,
        task_mask=None,
        
        
    ):
        
        output = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            output_hidden_states=True,
            return_dict=True,
        )[1]
   
        batch_size = output.shape[0]
        all_preds = torch.zeros((batch_size, self.num_labels)).to(output.device)
    
        for i in range(self.num_heads):
            
            cur_out = output * (task_mask == i)   
            
            preds = self.classifier_heads[i](cur_out)
            # Set the fields for this classifier
            all_preds += preds * (task_mask == i)
            
        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(all_preds, labels.view(-1).long())

        
        return SequenceClassifierOutput(
            loss=loss,
            logits=all_preds,
        )
    
class BertWithMeta(BertPreTrainedModel):
    
    def __init__(self, config, num_meta):
        super().__init__(config)
        
        self.num_labels = config.num_labels
                
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        
        total_size = config.hidden_size + num_meta
        self.classifier_a = nn.Linear(config.hidden_size + num_meta, config.hidden_size + num_meta)
        self.classifier_b = nn.Linear(config.hidden_size + num_meta, config.num_labels)

        self.init_weights()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        labels=None,
        metadata=None,
        
        
    ):
        
        output = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            output_hidden_states=True,
            return_dict=True,
        )[1]
   
        combined_output = torch.cat([output, metadata], axis=1)
    
        intermediate_output = self.classifier_a(combined_output)
        all_preds = self.classifier_b(intermediate_output)

            
        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(all_preds, labels.view(-1).long())

        
        return SequenceClassifierOutput(
            loss=loss,
            logits=all_preds,
        )