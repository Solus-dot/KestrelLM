from datasets import load_dataset
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.decoders import ByteLevel as ByteLevelDecoder
from pathlib import Path

from config import VOCAB_SIZE


dataset = load_dataset(
    "roneneldan/TinyStories",
    split="train",
)

tokenizer = Tokenizer(
    BPE(unk_token="<unk>")
)

tokenizer.pre_tokenizer = ByteLevel(add_prefix_space=False)
tokenizer.decoder = ByteLevelDecoder()

trainer = BpeTrainer(
    vocab_size=VOCAB_SIZE,
    special_tokens=[
        "<pad>",
        "<unk>",
        "<bos>",
        "<eos>",
    ],
)

tokenizer.train_from_iterator(
    dataset["text"],
    trainer=trainer,
)

Path("tokenizer").mkdir(exist_ok=True)
tokenizer.save("tokenizer/tokenizer.json")

print("Tokenizer trained.")
print("Vocabulary size:", tokenizer.get_vocab_size())
