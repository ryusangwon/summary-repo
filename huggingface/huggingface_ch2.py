# Batch Size: number of sequence
# Hidden Size: dimension of model input (768 for base, more than 3072 for large)

import torch
from transformers import AutoTokenizer, AutoModel, AutoModelForSequenceClassification
from transformers import BertConfig, BertModel

text = "Learn how to use transformer for summarization"
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
model = AutoModel.from_pretrained("bert-base-uncased")
inputs = tokenizer(text, return_tensors="pt") # no return_tensors will return list
outputs = model(**inputs)
logits = outputs[0] # To use feature
logits = outputs["last_hidden_state"]

print("input:", inputs)
print("text len:", len(text))
print("token len: ", len(inputs))
print("logit shape:", logits.shape)
print(inputs["input_ids"])
# decoded_string = tokenizer.decode(inputs["input_ids"])
# print("decoded:", decoded_string)

# using AutoModelForSequenceClassification (not AutoModel). It classify positive/negative
model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased")
outputs = model(**inputs)
print("logit shape", outputs.logits.shape)
print("logit", outputs.logits)
predictions = torch.nn.functional.softmax(outputs.logits, dim=-1) # (negative, positive)
print(predictions)

sequences = ["Hello!", "Cool.", "Nice!"]
encoded_sequence = tokenizer(sequences)
print(encoded_sequence)
