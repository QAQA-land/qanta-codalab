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
    vocab, stoi_, vectors, pad_index, unk_index = DanEmbedding.load_pretrained_weights(PRETRAINED_FP)
    dan = DanEmbedding(pretrained_weights=vectors)
    print('len vocab', len(vocab))
    #print(stoi_)
    print('weights shape', vectors.shape)
    print(dan.embed_dim)

@cli.command()
def dataloader():
    vocab, stoi_, vectors, pad_index, unk_index = DanEmbedding.load_pretrained_weights(PRETRAINED_FP)
    dataset = QuizBowlDataset(guesser_train=True)
    qs, pages, _ = dataset.training_data()
    dataset = TorchQBData(qs, pages, stoi_, pad_index, unk_index)
    print('# qs', len(dataset.qs))
    print('# ans', len(dataset.answers))
    print('ans examples:', dataset.answers[:5])

    train_dataloader = DataLoader(dataset, batch_size = 2, shuffle=True, num_workers=1, 
                                collate_fn=TorchQBData.collate)
    print('* GETTING BATCHES')
    for batch in train_dataloader:
        print('\n\n')
        print(batch)
        break

@cli.command()
def forward():
    vocab, stoi_, vectors, pad_index, unk_index = DanEmbedding.load_pretrained_weights(PRETRAINED_FP)

    dataset = QuizBowlDataset(guesser_train=True)
    qs, pages, _ = dataset.training_data()
    dataset = TorchQBData(qs, pages, stoi_, pad_index, unk_index)
    print('# qs', len(dataset.qs))
    print('# ans', len(dataset.answers))
    print('ans examples:', dataset.answers[:5])

    train_dataloader = DataLoader(dataset, batch_size = 5, shuffle=True, num_workers=1, 
                                collate_fn=TorchQBData.collate)

    dan_embed = DanEmbedding(pretrained_weights=vectors)
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
        answers = [dataset.i_to_ans[i] for i in indices.numpy()]
        print('predicted', answers)
        print('actual   ',[dataset.i_to_ans[i] for i in labels.numpy()])
        break

@cli.command()
def run():
    vocab, stoi_, vectors, pad_index, unk_index = \
        DanEmbedding.load_pretrained_weights(PRETRAINED_FP)

    dataset = QuizBowlDataset(guesser_train=True)

    guesser = DanGuesser(self, 
                 dataset.answers,
                 pretrained_weights=PRETRAINED_FP,
                 batch_size=5,
                 max_epochs=1,
                 grad_clip=5,
                 lr = 0.01,
                 patience=100,
                 embed_dim=None,
                 vocab_size=None,
                 n_hidden_units=50,
                 nn_dropout=0.5)

    tr_qs, tr_pages, _ = dataset.training_data()
    te_qs, te_pages, _ = dataset.test_data()

    
    # N_SAMPLE = 10
    # tr_qs, tr_pages = tr_qs[:N_SAMPLE], tr_pages[:N_SAMPLE]
    # te_qs, te_pages = te_qs[:N_SAMPLE], te_pages[:N_SAMPLE]

    

    train_dataset = TorchQBData(tr_qs, tr_pages, stoi_, pad_index, unk_index)
    val_dataset = TorchQBData(te_qs, te_pages, stoi_, pad_index, unk_index)

    guesser.train(train_dataset, val_dataset)
    print(tr_pages)
    print(te_pages)

if __name__ == '__main__':
    cli()
