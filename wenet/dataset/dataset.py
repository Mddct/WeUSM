import functools
from typing import Callable, List, Optional
from torch._C import _functionalization_reapply_views_tls
from torch.nn import functional
from torch.utils.data import IterDataPipe
import torchdata
import torch
import yaml
from torchdata.datapipes import functional_datapipe

from torchdata.datapipes.iter import FileOpener

from functools import partial


@functional_datapipe("try_map")
class TryMapIterDataPipe(IterDataPipe):

    def __new__(cls,
                datapipe: IterDataPipe,
                fn: Callable,
                input_col=None,
                output_col=None) -> None:

        def try_fn(x):
            try:
                return fn(x)
            except Exception as e:
                # eg: some error
                return None

        return datapipe.map(try_fn, input_col=input_col,
                            output_col=output_col)  # type: ignore


# TODO(Mddct): bpe/wpe tokenize
def tokenize_space(data, symbol_table, UNK='<unk>'):
    """ Decode text by space

        Args:
            data: tuple (filename, text)
            symbol_table: dict
        Returns:
            tuple: (filename, text, tokens, label)
    """
    filename, text = data
    tokens = text.split(" ")

    ids = [
        symbol_table[token] if token in symbol_table else symbol_table[UNK]
        for token in tokens
    ]
    return (filename, text, tokens, ids)


def filter(data, token_max_length=200, token_min_length=1):
    """ Filter text according to token length

    Args:
        data: tuple (filename, text, tokens, label)
    """
    ids = data[-1]
    return len(ids) <= token_max_length and len(ids) >= token_min_length


def padding_batch(data, padding_value=0):
    """ Padding the batch to tensor

    Args:
        data: batch [(filename, text, tokens, label), ...]
    """
    f, t, tk, l = [], [], [], []
    l = []
    for sample in data:
        filename, text, tokens, ids = sample
        l.append(torch.tensor(ids, dtype=torch.int, requires_grad=False))
        f.append(filename)
        t.append(text)
        tk.append(tokens)
    labels_tensor = torch.nn.utils.rnn.pad_sequence(l, batch_first=True)
    return f, t, tk, labels_tensor


# TODO(Mddct): use wenet
def read_symbol_table(symbol_table_file):
    symbol_table = {}
    with open(symbol_table_file, 'r', encoding='utf8') as fin:
        for line in fin:
            arr = line.strip().split()
            assert len(arr) == 2
            symbol_table[arr[0]] = int(arr[1])
    return symbol_table


def TextDataset(
    data_list_file,
    symbol_table,
    conf,
    prefetch=None,
    training=False,
) -> torchdata.datapipes.iter.IterDataPipe:
    dataset = FileOpener(data_list_file)

    # 1 shuffler list
    shuffle_conf = conf['shuffle_conf']
    if training:
        dataset = dataset.shuffle(
            buffer_size=shuffle_conf['list_shuffle_size'])

    # 2 shard for  each replica
    dataset = dataset.sharding_filter()

    # 3 shuffle for each file
    dataset = dataset.readlines()

    # 4 shuffle in file
    if training:
        dataset = dataset.shuffle(buffer_size=shuffle_conf['in_shuffle_size'])

    # 5 tokenize
    tokenize_fn = functools.partial(tokenize_space, symbol_table=symbol_table)
    dataset = dataset.map(tokenize_fn)

    # 6 filter
    filter_conf = conf.get('text_filter_conf', {})
    filter_fn = functools.partial(filter, **filter_conf)
    dataset = dataset.filter(filter_fn)

    # 6 bucket
    bucket_conf = conf.get('bucket_conf', {})
    batch_size = bucket_conf['batch_size']
    # bucket = [10, 30, 50, 100]
    bucket = bucket_conf.get('bucket', [])
    sort_fn = functools.partial(sorted, key=lambda data: len(data[-1]))
    dataset = dataset.bucketbatch(
        batch_size=batch_size,
        drop_last=True if training else False,
        batch_num=batch_size,
        bucket_num=len(bucket) + 1,
        sort_key=sort_fn,
    )
    # 7 pad batch
    dataset = dataset.collate(padding_batch)
    if prefetch is not None:
        dataset = dataset.prefetch(prefetch)
    return dataset


symbol_table = read_symbol_table("unit.txt")
config = 'test.yaml'
with open(config, 'r') as fin:
    configs = yaml.load(fin, Loader=yaml.FullLoader)
dataset = TextDataset(["test.txt", "test.2.txt"],
                      symbol_table,
                      conf=configs['dataset_conf'])

for batch in dataset:
    key, text, tokens, labels = batch
    _, _, _, = key, text, tokens
    print(labels)
