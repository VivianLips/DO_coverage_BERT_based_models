#!pip install datasets

import os
import torch
import numpy as np
from datasets import load_dataset
from transformers import (AutoTokenizer, AutoModelForTokenClassification,
    TrainingArguments, Trainer)
from torch.utils.data import Dataset
from seqeval.metrics import accuracy_score, f1_score, precision_score, recall_score

# load tokenizer and model and define labels
num_labels = 3 
label_list = ["O", "B-Disease", "I-Disease"]

model = AutoModelForTokenClassification.from_pretrained("microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract",num_labels=num_labels)
tokenizer = AutoTokenizer.from_pretrained("microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract") # BioBERT

# load the datasets
ncbi = load_dataset("ncbi_disease")
bc5cdr = load_dataset("tner/bc5cdr")

# Remap the bc5cdr dataset tags to match the NCBI tags
def remap_tags(input):

  # extract tokens and tags
  tokens = input['tokens']
  tags = input['tags']

  # create an empty list to store the remapped tags in
  remap_tags = []

  # Loop over the tags and remap them
  for tag in tags:
    if tag == 2: 
      remap_tags.append(1) # if tag is 2 remap it to 1 (B-disease)
    elif tag == 3:
      remap_tags.append(2) # if tag is 3 remap it to 2 (I-disease)
    else:
      remap_tags.append(0) # remap tag to 0 (O-disease)
      
  return {'tokens': tokens, 'tags': new_tags}



# align labels with tokens
def align_tokenized_labels(text, labels, max_length=256):
    tokenized_text = tokenizer(
        text,
        truncation=True,
        padding="max_length",
        max_length=max_length,
        is_split_into_words=True
    )  

  # Create an empty list for aligned token labels
  aligned_labels = []

  # loop through the labels
  for i, label in enumerate(labels):

    # Align a word token to a word index
    word_id = tokenized_text.word_ids(batch_index=i)
    previous_word_id = None
    aligned_labels_id = []

    for id in word_id:
      if id is None: # [CLS] or [SEP]
        aligned_labels_id.append(-100) # default

      # start the first token of a word with the label of the word
      elif id != previous_word_id:
        aligned_labels_id.append(label[id])

      # repeat the words label for the other tokens
      else:
        if label[id] != 0:
          aligned_labels_id.append(label[id])

        # if original was 0 keep it 0
        else:
          aligned_label_ids.append(0)
          
    # save aligned labels
    aligned_labels.append(aligned_labels_id)

  # Add the labels to the tokenized words
  tokenized_text['labels'] = aligned_labels

  return tokenized_text

def split_dataset(dataset, label):
  return align_tokenized_labels(dataset['tokens'], dataset[label])

# Combine the dataset for the combined fine-tuning
def combine_datasets(dataset1, dataset2):
  results = {}
  for k in dataset1:
    results[k] = dataset1[k] + dataset2[k]
  return results  

# Change the dictionary like datasets into pytorch datasets so it can be used with DataLoader
class CreatePytorchDataset(Dataset):

    # initialize and encode the inputs from the tokenized dictionary
    def __init__(self, encodings):
        self.encodings = encodings

    # defines the number of examples in the dataset
    def __len__(self):
        return len(self.encodings["input_ids"])

    # Return the tokenized examples into tensors
    def __getitem__(self, idx):
        return {k: torch.tensor(v[idx]) for k, v in self.encodings.items()}

# Compute the metrics from training and validating the models
def compute_metrics(prediction):

    # ground truth labels
    labels = prediction.label_ids

    # predicted labels
    preds = np.argmax(prediction.predictions, axis=2)

    y_true, y_pred = [], []

    # loop over the true and predicted sequences
    for label_seq, pred_seq in zip(labels, preds):
        for label, pred_ in zip(label_seq, pred_seq):

            # exclude the ignore index
            if label != -100:

                # append predicted and true labels to the lists
                y_true.append(label_list[label])
                y_pred.append(label_list[pred_])
              
    accuracy = accuracy_score(y_true, y_pred)
    precisions = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def main():

  bc5cdr = bc5cdr.map(remap_tags)

  ncbi_train = split_dataset(ncbi['train'], 'ner_tags')
  bc5cdr_train = split_dataset(bc5cdr['train'], 'ner_tags')

  ncbi_validation = split_dataset(ncbi['validation'], 'ner_tags')
  bc5cdr_validation = split_dataset(bc5cdr['validation'], 'ner_tags')

  # For the combined fine-tuning combine the datasets
  combined_train = combine_datasets(ncbi_train, bc5cdr_train)
  combined_validation = combine_datasets(ncbi_validation, bc5cdr_validation)

  # define training and validation data as pytorch datasets
  training_data = CreatePytorchDataset(ncbi_train)  # change to BC5CDR or the Combined training data
  validation_data = CreatePytorchDataset(ncbi_validation)  # change to BC5CDR or the Combined validation data

  # Train (fine-tune) the model
  training_args = TrainingArguments(
    output_dir="./combined_ner_model",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=3e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=8,
    weight_decay=0.01,
    logging_dir="./logs",
    logging_steps=500,
    save_total_limit=2,
    report_to="none"
)

  trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=training_data,
    eval_dataset=validation_data,
    compute_metrics=compute_metrics
)

trainer.train()

# save the fine-tuned model and tokenizer
model.save_pretrained("./PubMedBERT_combined_ner")
tokenizer.save_pretrained("./PubMedBERT_combined_ner")    

if __name__ == "__main__":
    main()

