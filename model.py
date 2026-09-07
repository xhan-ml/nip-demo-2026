# model.py
from transformers import BertForSequenceClassification

def build_bert_model(model_path, num_labels, device):
    model = BertForSequenceClassification.from_pretrained(
        model_path,
        num_labels=num_labels
    )
    model = model.to(device)
    return model
