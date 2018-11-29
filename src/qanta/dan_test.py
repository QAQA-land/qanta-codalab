from qanta.dan import DanModel, DanEmbedding
from qanta.torch.torch_dataset import TorchQBData
from qanta.dataset import QuizBowlDataset
from torch.utils.data import DataLoader

import click

@click.group()
def cli():
    pass

def embeddings():
    dan = DanModel(pretrained_fp='embeddings/small_embeddings.txt')
    vocab, stoi, vectors = dan.load_pretrained_weights('embeddings/small_embeddings.txt')

    print(vocab)
    print(stoi)
    print(vectors)
    print(vectors.shape)

@cli.command()
def run():
    dan = DanModel(pretrained_fp='embeddings/small_embeddings.txt')

    dataset = QuizBowlDataset(guesser_train=True)
    dataset = TorchQBData(dataset, stoi = dan.stoi, pad_index=dan.pad_index)
    print('# qs', len(dataset.qs))
    print('# ans', len(dataset.answers))
    print('ans', dataset.answers[:10])

    train_dataloader = DataLoader(dataset, batch_size = 2, shuffle=True, num_workers=1, 
                                collate_fn=TorchQBData.collate)
    print('* GETTING BATCHES')
    for batch in train_dataloader:
        print('\n\n')
        print(batch)
        #print(batch)
        break
    

if __name__ == '__main__':
    cli()
