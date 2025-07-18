import json

# load the information for the disease ontology - concepts and synonyms
def load_disease_ontology(ontology):
  # create dictionaries to store concepts and synonyms
  concepts = {}
  synonyms = {}

  # read the ontology
  with open(ontology, 'r', encoding='utf-8') as f:
    current_concept = None

  # loop through the ontology lines
  for line in f:
    # strip the lines from whitespaces
    line = line.strip()

    # check for name; identifier for a concept
    if line.startswith("name:"):
      # split the line of name and take the word after that as concept
      current_concept = line.split("name:")[1]
      # convert to lowercase and delete whitespace
      current_concept = current_concept.strip().lower()

      # create a key for concepts witin the concept dictionary
      concept[current_concept] = set()

    # check if line starts with synonym
    elif line.startswith("synonym:") and current_concept:
      # take the word after \ as synonym
      synonym = line.splt("\"")[1]
      # convert to lowercase and delete whitespace
      synonym = synonym.split().lower()

      # add synonyms to the concepts in the dictionary
      concepts[current_concept].add(synonym)

      # map synonym to main concept
      synonyms[synonym] = current_concept

    return concepts, synonyms

def main():
  # open the JSON file with cleaned entities
  with open("/content/cleaned_unique_entities", "r", encoding="utf-8") as f:
      cleaned_list = json.load(f)
    
  # create dictionaries for concepts with entities, and synonyms with concepts and entities
  exact_match = {}
  synonym_match = {}

  # loop through the cleaned list and find matches
  for entity in cleaned_list:
    if entity in concept_dict:
      exact_match[entity] = entity
    elif entity in synonym_dict:
      synonym_matches[entity] = (entity, synonym_dict[entity])

  # Count matches
  num_exact_matches = len(exact_matches)
  num_synonym_matches = len(synonym_matches)
  
  print(num_exact_matches)
  print(num_synonym_matches)

  return exact_match, synonym_match

  
