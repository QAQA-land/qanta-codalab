from qanta.dan import DanModel, DanEmbedding, DanGuesser
from qanta.torch.torch_dataset import TorchQBData
from qanta.dataset import QuizBowlDataset
from torch.utils.data import DataLoader

import click

PRETRAINED_FP = 'embeddings/50000_embeddings.txt'
@click.group()
def cli():
    pass

@cli.command()
def pretrained():
    dan = DanEmbedding(pretrained_fp=PRETRAINED_FP)
    vocab, stoi, vectors = dan.load_pretrained_weights(PRETRAINED_FP)
    print('len vocab', len(vocab))
    print(stoi)
    print(vectors.shape)
    print(dan.embed_dim)
@cli.command()
def dataloader():
    dan_embed = DanEmbedding(pretrained_fp=PRETRAINED_FP)    
    dataset = QuizBowlDataset(guesser_train=True)
    dataset = TorchQBData(dataset, stoi = dan_embed.stoi, pad_index=dan_embed.pad_index)
    print('# qs', len(dataset.qs))
    print('# ans', len(dataset.answers))
    print('ans examples:', dataset.answers[:5])

    train_dataloader = DataLoader(dataset, batch_size = 2, shuffle=True, num_workers=1, 
                                collate_fn=TorchQBData.collate)
    print('* GETTING BATCHES')
    for batch in train_dataloader:
        print('\n\n')
        print(batch)
        #print(batch)
        break

@cli.command()
def forward():
    dan_embed = DanEmbedding(pretrained_fp=PRETRAINED_FP)    
    dataset = QuizBowlDataset(guesser_train=True)
    dataset = TorchQBData(dataset, stoi = dan_embed.stoi, pad_index=dan_embed.pad_index)
    print('# qs', len(dataset.qs))
    print('# ans', len(dataset.answers))
    print('ans examples:', dataset.answers[:5])

    train_dataloader = DataLoader(dataset, batch_size = 2, shuffle=True, num_workers=1, 
                                collate_fn=TorchQBData.collate)

    dan = DanModel(embedding=dan_embed, 
                    embed_dim=dan_embed.embed_dim, 
                    n_classes=dataset.n_answers)
    for batch in train_dataloader:
        print('* Forward')
        input_text = batch['text']
        lengths = batch['len']
        labels = batch['labels']
        out = dan(input_text, lengths)
        print(out)

        values, indices = out.max(1)
        print(indices)
        answers = [dataset.i_to_ans[i] for i in indices.numpy()]
        print(answers)
        print([dataset.i_to_ans[i] for i in labels.numpy()])
        break

@cli.command()
def run():
    dataset = QuizBowlDataset(guesser_train=True)
    guesser = DanGuesser(pretrained_fp=PRETRAINED_FP,
                         quizbowl_dataset=dataset,
                         n_training_samples=10,
                         max_epochs=100)
    print(guesser.torch_qb_data.n_answers)
    train_dataloader = DataLoader(guesser.torch_qb_data, 
                                batch_size = 20, 
                                shuffle=True, num_workers=1,
                                collate_fn=TorchQBData.collate)
    guesser.train()

if __name__ == '__main__':
    cli()
