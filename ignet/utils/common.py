from enum import Enum
from transformers import MPNetTokenizerFast as EncTokenizer
from transformers import MPNetModel as EncModel

ENC_MODEL_NAME = 'sentence-transformers/multi-qa-mpnet-base-dot-v1'
TOKENIZER = EncTokenizer.from_pretrained(ENC_MODEL_NAME)
CLS_TOK = TOKENIZER.cls_token_id
SEP_TOK = TOKENIZER.sep_token_id
PAD_TOK = TOKENIZER.pad_token_id

IG_TOK_NUM = 6
IG_TOKS = [f'<IG{i}>' for i in range(IG_TOK_NUM)]
TOKENIZER.add_special_tokens({'additional_special_tokens': IG_TOKS})
IG_TOKS = [TOKENIZER.vocab[t] for t in IG_TOKS]

MAX_CTX_LEN = 160
MAX_SENT_LEN = 96 - len(IG_TOKS)


class Label:
  RELEVANT = 0
  REPETITION = 1
  IRRELEVANT = 2


class Column:
  IDS = 'input_ids'
  MASK = 'attention_mask'
  CTX = 'ctx'
  CTX_MASK = CTX + '_mask'
  SENT = 'sent'
  SENT_MASK = SENT + '_mask'
  LABEL = 'label'


def encode_data_batch(data_batch):
  ctxs = TOKENIZER(data_batch[Column.CTX],
                   truncation=True,
                   padding='max_length',
                   max_length=MAX_CTX_LEN)
  sents = TOKENIZER(data_batch[Column.SENT],
                    truncation=True,
                    padding='max_length',
                    max_length=MAX_SENT_LEN)
  for sent, mask in zip(sents[Column.IDS], sents[Column.MASK]):
    sent[:] = IG_TOKS + sent
    mask[:] = [1] * len(IG_TOKS) + sent

  # rename columns
  ctxs[Column.CTX] = ctxs.pop(Column.IDS)
  ctxs[Column.CTX_MASK] = ctxs.pop(Column.MASK)
  sents[Column.SENT] = sents.pop(Column.IDS)
  sents[Column.SENT_MASK] = sents.pop(Column.MASK)
  return {**ctxs, **sents}
