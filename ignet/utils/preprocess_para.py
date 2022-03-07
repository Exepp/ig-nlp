import random
from .common import Column, Label

THREADS = 8


def process(dataset, sent1_col, sent2_col, label_col, label_th):

  def process_batch(data_batch):
    ctxs = data_batch[Column.CTX]
    sents = data_batch[Column.SENT]
    labels = data_batch[Column.LABEL]
    for i, (ctx, sent, label) in enumerate(zip(ctxs, sents, labels)):
      if random.getrandbits(1):
        ctxs[i] = sent
        sents[i] = ctx
      labels[i] = (Label.RELEVANT if label < label_th else Label.REPETITION)
    return data_batch

  cols_to_remove = set(dataset.features.keys())
  cols_to_remove -= {sent1_col, sent2_col, label_col}
  dataset = dataset.remove_columns(list(cols_to_remove))

  column_map = {sent1_col: Column.CTX, sent2_col: Column.SENT, label_col: Column.LABEL}
  for prev, new in column_map.items():
    if new not in dataset.features:
      dataset = dataset.rename_column(prev, new)

  return dataset.map(process_batch, batched=True, batch_size=512, num_proc=THREADS)