import torch
import torch.nn as nn
from transformers import BertModel


class BertTextClassifier(nn.Module):
    """
    自己组装BERT分类网络：
    基础BertModel + dropout + 手写Linear分类头
    """
    def __init__(self, model_name: str, num_classes: int, dropout_prob: float,hidden_size:int):
        super().__init__()

        # 加载原生BERT基础模型，不带分类头
        self.bert = BertModel.from_pretrained(
            model_name,
            # hidden_dropout_prob=0.2,
            # attention_probs_dropout_prob=0.2
        )

        self.dropout = nn.Dropout(dropout_prob)

        # bert输出的隐藏维度是768，输出类别15
        self.classifier = nn.Linear(in_features=hidden_size, out_features=num_classes)

    def forward(self, input_ids, attention_mask):
        """
        input_ids: [batch, seq_len]
        attention_mask: [batch, seq_len]
        返回 logits: [batch, num_classes]
        """
        # bert前向传播
        bert_out = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        cls_emb = bert_out.pooler_output

        out = self.dropout(cls_emb)
        logits = self.classifier(out)

        return logits
