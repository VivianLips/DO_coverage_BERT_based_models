from transformers import AutoModelForTokenClassification, AutoTokenizer
import json
import torch


# Creates full entities from labeled entity tokens
def extract_entities(tokens):
  # delete ## markers from the extraction process and covert token to string
  token = " ".join(tokens).replace(" ##", "")

  # set limit to max length 512
  inputs = tokenizer(token, truncation=True, max_length = 512, return_tensors='pt')

  # obtain the predictions for the tokens and classify them
  with torch.no_grad():
    outputs = model(**inputs)

  # obtain the best predicted class
  prediction = torch.argmax(outputs.logits, dim=2)

  # map each ID to the corresponding token
  tokens = tokenizer.convert_ids_to_tokens(inputs['inputs_ids'][0])
  prediction_labels = prediction[0].tolist()

  entities = []
  currennt_entity = []
  
  # Loop through tokens and labels and merge the tokens
  for token, label in zip(tokens, prediction_labels):
    # Ignore special tokens used for the models
    if token in ["[CLS]", "[SEP]"]: 
      continue

    # identify beginning of entities (1)
    if label == 1:
      # the token is beginning label but it stars with ## so it is actually a continuation
      if token.startswith("##") and current_entity:
        # add it to the current entity
        current_entity[-1] += token[2:]
      else:
        if current_entity:
          # save current entity and start a new one
          entities.append(" ".join(current_entity))
        # define new current entity
        current_entity = [token]


    # Check for inside labels (2)
    if label == 2:
      # if the token starts with ## add it to the previous token
      if token.startswith("##") and current_entity:
        current_entity[-1] += token[2:]

      # if normal add it to current token
      else:
        current_entity.append(token)

    # if it is an outside token add as " "
    else:
      if current_entity:
        entities.append(" ".join(current_entity))

    # return the list of entities
    return entities

def open_json_file(file):
  # read the preprocessed corpus
  with open(file, 'r') as f:
    corpus = json.load(f)
  return corpus


def main()


# define the directory with fine-tuned model 
output_dir = '\content\model'

# read the corpus
preprocessed_corpus = open_json_file(output_dir)

# Load the model and tokenizer from the fine-tuned model 
model = AutoModelForTokenClassification.from_pretrained(output_dir)
tokenizer = AutoTokenizer.from_pretrained(output_dir)

# create a list of extracted entities to store the entities
extracted_entities = []

# loop over the tokens in the preprocessed corpus
for tokens in preprocessed_corpus:
  entities = extract_entities(tokens)

  # add entities to the list
  extracted_entities.append(entities)

# obtain unique entities
unique_entities = set(entity for entities in all_extracted_entities for entity in entities)

return unique entities

if __name__ == "__main__":
  main()
