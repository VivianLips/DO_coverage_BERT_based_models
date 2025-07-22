nltk.download('wordnet')
nltk.download('omw-1.4')

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

# Read a corpus of text
def open_json_file(file):
  with open(file, 'r') as f:
    corpus = json.load(f)
  return corpus


# Clean list of entities
def clean_entity_list(entities):
  # Create an empty list to store the cleaned entities
  cleaned_list = []
  lemmatizer = WordNetLemmatizer()

  # Loop over the entities
  for entity in entities:
    # lowercase and strip white spaces
    entity = entity.lower().strip()

    # if the entity contains ## somewhere, skip it (bad token alignment)
    if '##' in entity:
      continue

    # Remove noise in tokens - Only use if needed
    #entity = re.sub(r"[\[\]\(\)\\\/]", " ", entity) # brackets/slashes
    #entity = re.sub(r"[^\w\s\-\+\.\']", "", entity) # other special characters

    # fix spacing between ' and -
    entity = re.sub(r"\s*'\s*", "'", entity)
    entity = re.sub(r"\s*\.\s*", ".", entity)
    entity = re.sub(r"\s*[-]\s*", "-", entity)
    entity = re.sub(r"\s*[\+]\s*", "+", entity)

    # lemmatize last words from entities to create singular forms from plural forms
    last_word = entity.split()
    if last_word:
      last_word[-1] = lemmatizer.lemmatize(last_word[-1], pos='n')
      entity = " ".join(last_word)

    if entity:
      cleaned_list.append(entity)

    return cleaned_list
      
# save cleaned entity list in a JSON file
def save_cleaned_entities(output_path, cleaned_entities):
  with open(output_path, "w", encoding="utf-8") as f:
    json.dump(cleaned_entities, f, indent=2)
  return cleaned_entities


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

# Clean the list of entities
cleaned_entities = clean_entity_list(unique_entities)

output_path = "/content/cleaned_unique_entities"

# save the list to a JSON file
cleaned_entities_list = save_cleaned_entities(output_path, cleaned_entities)

return cleaned_entities_list

if __name__ == "__main__":
  main()
