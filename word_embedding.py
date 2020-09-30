# Author: Robert Guthrie

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import json
import os
import numpy as np

torch.manual_seed(1)

with open(os.path.join(opt.dataset_path, 'json_dataset', 'objects.json'), 'r') as f:
	objects = json.load(f)

word_to_ix = {}
for i, obj in enumerate(objects):
	word_to_ix[obj] = i

glove = {}
vocab = len(objects)
matrix_len = vocab
weights_matrix = np.zeros((matrix_len, 300))
with open(opt.glove_path, 'rb') as f:
	for l in f:
		line = l.decode().split()
		word = line[0]
		vect = np.array(line[1:]).astype(np.float)
		glove[word] = vect
for i, obj in enumerate(objects):
	try:
		weights_matrix[word_to_ix[obj]] = glove[obj]
	except KeyError:
		weights_matrix[word_to_ix[obj]] = np.random.normal(
			scale=0.6, size=(300, ))
weights_matrix = torch.Tensor(weights_matrix)


word_to_ix = {"hello": 0, "world": 1}
embeds = nn.Embedding(2, 5)  # 2 words in vocab, 5 dimensional embeddings
lookup_tensor = torch.tensor([word_to_ix["hello"]], dtype=torch.long)
hello_embed = embeds(lookup_tensor)
print(hello_embed)