from collections import defaultdict 
import os
import random 

from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import KFold
from transformers import Trainer, TrainingArguments

def do_metrics(res):
    predictions = res.predictions[0]
    a = f1_score((predictions > 0.5).reshape(-1), res.label_ids, average="macro")
    b = f1_score((predictions > 0.5).reshape(-1), res.label_ids, average="micro")
    c = f1_score((predictions > 0.5).reshape(-1), res.label_ids, average=None)[1]
    d = accuracy_score((predictions > 0.5).reshape(-1), res.label_ids)

    return {'macro_f1': a, 'micro_f1': b, 'pos_f1': c, 'accuracy': d}

def split_by_doc_id(data, num_splits=5):
    
    documents = data.doc_id.unique()
    
    kf = KFold(num_splits, shuffle=True)
    
    for train_idx, test_idx in kf.split(documents):
        
        train_docs = documents[train_idx]
        random.shuffle(train_docs)
        test_docs = documents[test_idx]
        random.shuffle(test_docs)
        
        train_data = data[data.doc_id.isin(train_docs)]
        test_data = data[data.doc_id.isin(test_docs)]
        
        yield train_data.to_dict(orient='records'), test_data.to_dict(orient='records')
    

def run_full_cv(data, model_class, model_args, training_args, averaged=False):
    
    final_results = []
    for train_data, test_data in split_by_doc_id(data):
        
        model = model_class(**model_args)
        
        temp_path = os.path.expanduser('~/temp_cv_path')
        args = TrainingArguments(**training_args, disable_tqdm=True, logging_steps=10000, save_steps=10000, output_dir=temp_path)
        
        trainer = Trainer(model=model, train_dataset=train_data, eval_dataset=test_data, compute_metrics=do_metrics, args=args)
        
        trainer.train()
        
        res = trainer.predict(test_data)
        final_results.append(res)
        print("Finished round", res.metrics)
    
    if averaged:
        # Only keep averaged results
        final_averages = defaultdict(list)
        
        for entry in final_results:
            for k, v in entry.metrics.items():
                final_averages[k].append(v)
        final_results = {k: sum(v) / len(v) for k, v in final_averages.items()}

    return final_results
