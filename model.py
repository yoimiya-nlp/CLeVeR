import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import RobertaModel


def info_nce_loss(query, key, temperature=0.1):
    batch_size = query.size(0)
    labels = torch.arange(batch_size).to(query.device)

    # Normalize features
    query_norm = F.normalize(query, dim=1)
    key_norm = F.normalize(key, dim=1)

    # Compute logits
    logits = torch.mm(query_norm, key_norm.t()) / temperature

    # Compute loss
    loss = F.cross_entropy(logits, labels)
    return loss


class CrossAttention(nn.Module):
    def __init__(self, att_hidden_size):
        super(CrossAttention, self).__init__()
        self.query_layer = nn.Linear(att_hidden_size, att_hidden_size)
        self.key_layer = nn.Linear(att_hidden_size, att_hidden_size)
        self.value_layer = nn.Linear(att_hidden_size, att_hidden_size)
        self.attention = nn.MultiheadAttention(att_hidden_size, num_heads=8)

    def forward(self, query, key, value):       # default: batch_first=False
        # Add an extra dimension for the number of heads
        query = self.query_layer(query).unsqueeze(0)  # shape: (1, batch_size, hidden_size), 1 is target seq_len
        key = self.key_layer(key).permute(1, 0, 2)  # shape: (seq_len, batch_size, hidden_size)
        value = self.value_layer(value).permute(1, 0, 2)  # shape: (seq_len, batch_size, hidden_size)

        # Multi-head Attention expects (seq_len, batch_size, hidden_size) for key and value
        attn_output, _ = self.attention(query, key, value)
        return attn_output.squeeze(0)  # shape: (batch_size, hidden_size)


class FFN(nn.Module):
    def __init__(self, args):
        super(FFN, self).__init__()
        self.dense_1 = nn.Linear(args.hidden_size, args.hidden_size)
        self.dropout = nn.Dropout(0.3)
        self.dense_2 = nn.Linear(args.hidden_size, args.hidden_size)

    def forward(self, x):
        x = self.dense_1(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.dense_2(x)
        return x


class DescriptionAdapter(nn.Module):
    def __init__(self, args):
        super(DescriptionAdapter, self).__init__()
        self.ffn = FFN(args)
        self.layer_norm = nn.LayerNorm(args.hidden_size)

    def forward(self, x):
        x = self.ffn(x)
        x = self.layer_norm(x)
        return x


class CodeAdapter(nn.Module):
    def __init__(self, args):
        super(CodeAdapter, self).__init__()
        self.self_attention = nn.MultiheadAttention(args.hidden_size, num_heads=8)
        self.ffn = FFN(args)
        self.layer_norm = nn.LayerNorm(args.hidden_size)

    def forward(self, x):
        y, _ = self.self_attention(x.unsqueeze(0), x.unsqueeze(0), x.unsqueeze(0))
        y = y.squeeze(0)
        y = self.ffn(y)
        x = x + y
        x = self.layer_norm(x)
        return x


class CodeEncoder(nn.Module):
    def __init__(self, args):
        super(CodeEncoder, self).__init__()
        self.encoder = RobertaModel.from_pretrained(args.pretrain_code_model_name)
        self.code_adapter = CodeAdapter(args)
        self.cross_attention = CrossAttention(att_hidden_size=args.hidden_size)
        self.description_classifier = FFN(args)

    def forward(self, code_input, code_attention, des_query=None, flag=None):
        outputs = self.encoder(input_ids=code_input, attention_mask=code_attention)
        code_hidden_states = outputs.last_hidden_state
        cls = code_hidden_states[:, 0, :]  # return representation of CLS token
        cls = self.code_adapter(cls)
        if flag == "train":
            code_refined_representation = self.cross_attention(des_query, code_hidden_states, code_hidden_states)
        else:
            code_refined_representation = None

        if flag == "train":
            loss_function = nn.MSELoss()
            desc_cross_att_logits = self.description_classifier(cls)
            description_loss = loss_function(desc_cross_att_logits, code_refined_representation)

            lamda_1 = 0.2
            vulnerability_code_representation = lamda_1 * cls + (1 - lamda_1) * code_refined_representation
            return vulnerability_code_representation, description_loss
        else:
            code_refined_representation = self.description_classifier(cls)
            lamda_1 = 0.2
            vulnerability_code_representation = lamda_1 * cls + (1 - lamda_1) * code_refined_representation
            return vulnerability_code_representation, None


class DescriptionEncoder(nn.Module):
    def __init__(self, args):
        super(DescriptionEncoder, self).__init__()
        self.encoder = RobertaModel.from_pretrained(args.pretrain_text_model_name)
        self.description_adapter = DescriptionAdapter(args)

    def forward(self, input_ids, attention_mask):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        desc_hidden_states = outputs.last_hidden_state
        cls = desc_hidden_states[:, 0, :]  # return representation of CLS token
        cls = self.description_adapter(cls)
        return cls


class ContrastiveModel(nn.Module):
    def __init__(self, args):
        super(ContrastiveModel, self).__init__()
        self.code_encoder = CodeEncoder(args)
        self.desc_encoder = DescriptionEncoder(args)

    def forward(self, func_input_ids, func_attention_mask, description_input_ids=None, description_attention_mask=None,
                source_input_ids=None, source_attention_mask=None, sink_input_ids=None, sink_attention_mask=None, flag=None):

        if flag == "train":
            reason_cls = self.desc_encoder(input_ids=description_input_ids, attention_mask=description_attention_mask)
            source_cls = self.desc_encoder(input_ids=source_input_ids, attention_mask=source_attention_mask)
            sink_cls = self.desc_encoder(input_ids=sink_input_ids, attention_mask=sink_attention_mask)
            lamda_0 = 0.25
            description_representation = lamda_0 * source_cls + lamda_0 * sink_cls + (1 - 2 * lamda_0) * reason_cls
        elif flag == "test":
            description_representation = self.desc_encoder(input_ids=description_input_ids, attention_mask=description_attention_mask)
        else:
            description_representation = None

        vulnerability_code_representation, description_loss = self.code_encoder(func_input_ids, func_attention_mask,
                                                                                description_representation, flag)
        if flag == "train":
            info_loss = info_nce_loss(vulnerability_code_representation, description_representation)
            alpha = 0.7
            loss = info_loss + alpha * description_loss
            return loss
        elif flag == "vul":
            return vulnerability_code_representation
        else:
            return vulnerability_code_representation, description_representation


class LinearProbe(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(LinearProbe, self).__init__()
        self.fc = nn.Linear(input_dim, num_classes)

    def forward(self, x):
        return self.fc(x)

