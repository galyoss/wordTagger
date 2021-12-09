import matplotlib.pyplot as plt  # for plotting
import pandas as pd  # only to show some data in a nice table
import torch

from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import WhitespaceSplit
from tokenizers.trainers import WordLevelTrainer

from datasets import load_dataset
from torch import nn, optim
from torch.utils.data import DataLoader
from tqdm.notebook import tqdm  # progress bar

SEED = 5782

import random
from numpy import random as nprnd

random.seed(SEED)
nprnd.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

pd.set_option('display.max_colwidth', None)
dataset = load_dataset(
   'universal_dependencies', 'mr_ufal')
dataset.set_format(type="pandas", columns=["text", "tokens", "upos"])
dataset['validation'][:10]

val_tags = dataset['validation'].features['upos'].feature.names
[f'{i:2}: {p}' for (i,p) in enumerate(val_tags)]

### for exercise 1 ###

val_tags = dataset['validation'].features['upos'].feature.names
trn_tags = dataset['train'].features['upos'].feature.names
test_tags = dataset['test'].features['upos'].feature.names
assert(val_tags==trn_tags==test_tags)

PAD_ID = 0

def map_instance_to_whitespace_tokenizable_text(inst) -> str:
    return " ".join(inst['tokens'])

def make_tokenizers():
    dataset = load_dataset("universal_dependencies", "mr_ufal", split="train")
    tokenizer = Tokenizer(WordLevel(unk_token="<UNK>"))
    tokenizer.pre_tokenizer = WhitespaceSplit()
    trainer = WordLevelTrainer(special_tokens=["<PAD>", "<UNK>"])
    tokenizer.train_from_iterator([map_instance_to_whitespace_tokenizable_text(i) for i in dataset],
                                  trainer=trainer,
                                  length=len(dataset))
    tokenizer.enable_padding(pad_id=PAD_ID, pad_token="<PAD>")

    tag_tokenizer = Tokenizer(WordLevel(vocab={str(i): i for i in range(len(val_tags)+1)}))
    tag_tokenizer.enable_padding(pad_id=len(val_tags), pad_token=str(len(val_tags)))
    return tokenizer, tag_tokenizer

tokenizer, tag_tokenizer = make_tokenizers()

tokenizer.save("ud-mr-tokenizer.json", pretty=True) #TODO: export the json
tag_tokenizer.save("ud-mr-tag-tokenizer.json", pretty=True)

print(tokenizer.get_vocab_size())
print(tag_tokenizer.get_vocab_size())

device = "cuda" if torch.cuda.is_available() else "cpu"


class PosTagger(nn.Module):
    def __init__(self,
                 embedding_dim: int,
                 hidden_size: int,
                 num_tags: int,
                 num_layers: int) -> None:
        super().__init__()

        ### for exercise 2.1 ###

        self.embedding = nn.Embedding(tokenizer.get_vocab_size(),
                                      embedding_dim,
                                      padding_idx=tokenizer.padding["pad_id"])
        self.lstm = nn.LSTM(input_size=embedding_dim,
                            hidden_size=hidden_size,
                            num_layers=num_layers,
                            batch_first=True,
                            bidirectional=True)
        self.fc = nn.Linear(hidden_size * 2, tag_tokenizer.get_vocab_size())

    def forward(self, x) -> torch.Tensor:
        embs = self.embedding(x)
        output, (hidden, cell) = self.lstm(embs)
        return torch.sigmoid(self.fc(output))


def train(model: PosTagger,
          optimizer: optim.Optimizer,
          loss_fn: nn.CrossEntropyLoss,
          dataloader: DataLoader) -> dict:
    model.train()
    total = 0
    correct = 0
    for batch in dataloader:
        optimizer.zero_grad()
        sentences = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)
        probs = model(sentences)
        labels_for_loss = translate_label_vector_for_lossfunc(labels.view(-1))
        loss = loss_fn(probs.permute(0, 2, 1), labels)
        loss.backward()
        optimizer.step()
        ###

        preds = probs.argmax(dim=2)

        ### for exercise 3.2 ###

        pass

        ###

    return correct / total


def translate_label_vector_for_lossfunc(labels):
    output= []
    for label in labels:
        output.append([1 if i == label else 0 for i in range(tag_tokenizer.get_vocab_size())])
    return torch.FloatTensor(output)

def evaluate(model: PosTagger,
             dataloader: torch.utils.data.DataLoader) -> dict:
    model.eval()
    total = 0
    correct = 0
    with torch.no_grad():  # operations done in this block will not contribute to gradients
        for batch in tqdm(dataloader, desc="Evaluation"):
            sentences = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)
            probs = model(sentences)
            preds = probs.argmax(dim=2)

            ### for exercise 3.2 ###

            ### (same code from train() should work)

    return correct / total

BATCH_SIZE = 16
EMB_DIM = 100
HIDDEN_DIM = 128
NUM_LAYERS = 2
EPOCHS = 5

def deep_stringify(x):
    if type(x) == int:
        return str(x)
    return [deep_stringify(a) for a in x]

dataset = load_dataset("universal_dependencies", "mr_ufal")
dataset = dataset.map(lambda ins: {
    "input_ids": [e.ids for e in tokenizer.encode_batch(ins['tokens'],
                                                        is_pretokenized=True)],
    "labels": [e.ids for e in tag_tokenizer.encode_batch(deep_stringify(ins['upos']),
                                                        is_pretokenized=True)],
}, batched=True, batch_size=BATCH_SIZE)
dataset.set_format(type="torch", columns=["input_ids", "labels"])

train_dataloader = DataLoader(dataset["train"], batch_size=BATCH_SIZE)
val_dataloader = DataLoader(dataset["validation"], batch_size=BATCH_SIZE)
test_dataloader = DataLoader(dataset["test"], batch_size=BATCH_SIZE)

#=======HERE "main" begins

model = PosTagger(embedding_dim=EMB_DIM,
                  hidden_size=HIDDEN_DIM,
                  num_layers=NUM_LAYERS,
                  num_tags=len(val_tags)).to(device)
optimizer = optim.Adam(model.parameters())

### for exercise 4

loss_fn = torch.nn.CrossEntropyLoss() #is it that???

###

train_accuracies = []
validation_accuracies = []
for epoch in range(EPOCHS):
    train_acc = train(model, optimizer, loss_fn,
                          train_dataloader)
    val_acc = evaluate(model, val_dataloader)
    print(f"Epoch {epoch + 1}:")
    print(f"Training Accuracy: {100 * train_acc:.2f}%")
    print(f"Validation Accuracy: {100 * val_acc:.2f}%")
    train_accuracies.append(train_acc)
    validation_accuracies.append(val_acc)

plt.title("Accuracy by Epoch")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.plot(train_accuracies, label="Training")
plt.plot(validation_accuracies, label="Validation")
plt.legend();