import os
import torch 
from transformers import get_linear_schedule_with_warmup
from transformers import AutoModel, AutoTokenizer, AutoConfig
from sentence_transformers import SentenceTransformer
from torch.optim.lr_scheduler import StepLR

BERT_CLASS = {
    "distilbert": 'distilbert-base-uncased', 
    "bert": 'bert-base-uncased',
    "roberta": 'roberta-base',
}

SBERT_CLASS = {
    "distilbert": 'distilbert-base-nli-stsb-mean-tokens',
}


def get_optimizer(model, args):
    optimizer = torch.optim.Adam([
        {'params': model.bert.parameters()}, 
        {'params': model.contrast_head.parameters(), 'lr': args.lr*args.lr_scale},
        {'params': model.adapt.parameters(), 'lr': args.lr * args.lr_scale},
        {'params': model.prob.parameters(), 'lr': args.lr * args.lr_scale},
    ], lr=args.lr)

    return optimizer


def get_bert(args):
    
    if args.use_pretrain == "SBERT":
        bert_model = get_sbert(args)
        tokenizer = bert_model[0].tokenizer
        model = bert_model[0].auto_model
        print("..... loading Sentence-BERT !!!")
    else:
        config = AutoConfig.from_pretrained(BERT_CLASS[args.bert])
        model = AutoModel.from_pretrained(BERT_CLASS[args.bert], config=config)
        tokenizer = AutoTokenizer.from_pretrained(BERT_CLASS[args.bert])
        print("..... loading plain BERT !!!")
        
    return model, tokenizer


def get_sbert(args):
    # sbert = SentenceTransformer(SBERT_CLASS[args.bert])
    sbert = SentenceTransformer('pretrain-model-M3')
    return sbert








