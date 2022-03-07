from pathlib import Path
from utils import preprocess_para, preprocess_wiki
from utils.common import encode_data_batch
from datasets import load_dataset, concatenate_datasets

DATASET_NAME = 'ig_dataset'
DATASET_PATH_TRAIN = (Path(DATASET_NAME) / 'train').resolve()
DATASET_PATH_EVAL = (Path(DATASET_NAME) / 'eval').resolve()

TEXT_DATASETS = [('wikipedia', '20200501.en', ('text'))]
PARA_DATASETS = [('glue', 'mrpc', ('sentence1', 'sentence2', 'label', 1)),
                 ('paws', 'labeled_final', ('sentence1', 'sentence2', 'label', 1))]

THREADS = 8


def mk_split(split, dst_path):
  main_dataset = None
  for path, name, args in PARA_DATASETS:
    dataset = load_dataset(path, name, split=split)
    dataset = preprocess_para.process(dataset, *args)
    if main_dataset is None:
      main_dataset = dataset
    else:
      main_dataset = concatenate_datasets([main_dataset, dataset])
  main_dataset = main_dataset.map(encode_data_batch, batched=True, batch_size=512, num_proc=THREADS)
  main_dataset.save_to_disk(dst_path)


def main():
  mk_split('train', DATASET_PATH_TRAIN)
  mk_split('validation', DATASET_PATH_EVAL)


if __name__ == '__main__':
  main()
