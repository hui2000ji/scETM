from transformers import RobertaConfig
config = RobertaConfig(
    vocab_size=1000,
    max_position_embeddings=1001,
    num_attention_heads=12,
    num_hidden_layers=6,
    type_vocab_size=1,
)

from transformers import RobertaTokenizerFast
tokenizer = RobertaTokenizerFast.from_pretrained('./promotor/bert')


from transformers import RobertaForMaskedLM
model = RobertaForMaskedLM(config=config)


# from transformers import LineByLineTextDataset
# dataset = LineByLineTextDataset(
#     tokenizer=tokenizer,
#     file_path="./promotor/promotor_seq.txt",
#     block_size=None,
# )


# from transformers import DataCollatorForLanguageModeling
# data_collator = DataCollatorForLanguageModeling(
#     tokenizer=tokenizer, mlm=True, mlm_probability=0.15
# )


# from transformers import Trainer, TrainingArguments
# training_args = TrainingArguments(
#     output_dir="./promotor/log",
#     overwrite_output_dir=True,
#     num_train_epochs=10,
#     per_device_train_batch_size=32,
#     save_steps=1330,
#     save_total_limit=3,
# )

# trainer = Trainer(
#     model=model,
#     args=training_args,
#     data_collator=data_collator,
#     train_dataset=dataset,
#     prediction_loss_only=True,
# )
# print(trainer.train())
# trainer.save_model("./promotor/log")

import torch
model.load_state_dict(torch.load('promotor/log/pytorch_model.bin'))

from transformers import pipeline
feature_extraction = pipeline(
    "feature-extraction",
    model=model,
    tokenizer=tokenizer,
    device=0
)

with open('promotor/promotor_seq.txt') as f:
    seqs = f.read().split('\n')
with open('promotor/promotor_gene.txt') as f:
    genes = f.read().split('\n')

from tqdm import tqdm
import numpy as np
batch_size = 32
n_genes = len(seqs)
avg = np.zeros((n_genes, 1000))
first = np.zeros_like(avg)

for start in tqdm(range(0, n_genes, batch_size), total=(n_genes - 1) // batch_size + 1):
    seq = seqs[start: start + batch_size]
    gene = genes[start: start + batch_size]

    result = np.array(feature_extraction(seq))  # [batch_size, n_tokens, n_hidden]
    avg[start: start + batch_size, :] = result.mean(1)
    first[start: start + batch_size, :] = result[:, 0, :]

import pickle
with open('promotor/avg.pkl', 'wb') as f:
    pickle.dump(avg, f)
with open('promotor/first.pkl', 'wb') as f:
    pickle.dump(first, f)