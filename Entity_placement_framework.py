import os
import requests
import torch
import torch.nn.functional as F
import pandas as pd
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from sentence_transformers import SentenceTransformer, util
from collections import defaultdict
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Install required libraries
#!pip install transformers sentence-transformers pandas --quiet 

# Load hthe model and tokenizer for NLI
tokenizer = AutoTokenizer.from_pretrained("textattack/roberta-base-MNLI")
nli_model = AutoModelForSequenceClassification.from_pretrained("textattack/roberta-base-MNLI")

# Function that defines natural language inference (NLI) classes for subclass relations
def determine_NLI_classes(entity, ontology_term):
  hypothesis = entity
  premise = ontology_term

  # Encode premise and hypothesis as a pair 
  inputs = tokenizer.encode(premise, hypothesis, return_tensors='pt', truncation = True)

  # pass the inputs through the NLI model and obain raw scores
  logits = nli_model(inputs)[0]

  # Convert logits to normalized outputs using SoftMax
  probabilities = F.softmax(logits, dim=1)

  # Define scores obtained from probabilities
  entailment_score = probabilities[0][2].item()
  contradiction_score = probabilities[0][0].item()
  neutral_score = probabilities[0][1].item()

  # Check entailment score and define classes
  if entailment_score > 0.6:
    return "Ontology is subclass - Entity is superclass"
  elif contradiction_score > 0.6:
    return "Unrelated (Manual Review)"
  elif neutral_score > 0.6:
    # Reverse the premise - hypothesis pair 
    reversed_inputs = tokenizer.encode(hypothesis, premise, return_tensors='pt', truncation=True)
    reversed_logits = nli_model(reversed_inputs)[0]
    reversed_probabilities = F.softmax(reversed_logits, dim=1)

    # Check entailment score and define classes
    reversed_entailment = reversed_probabilities[0][2].item()

    if reversed_entailment > 0.6:
      return "Entity is a subclass - Ontology is superclass"
  
  # Both entailment scores are too low, so we cannot identify the directionality of the relationship
  else:
    return "Relation unclear" 


# Create clusters of synonyms that are equivalent
def cluster_synonym(results, model, threshold=0.9):
  
    # Group extracted entities by their best ontology match
    synonym_to_entities = defaultdict(list)

    # Loop through the matching results
    for result in results:
       # Check for synonym suggestions
        if result['Recommendation'] == 'Add as synonym':
            synonym_to_entities[result['Best ontology match']].append(result['Extracted entity'])

    # Create an empty dictionary to store the synonym clusters in
    clusters_per_synonym = {}

    # Loop through the synonyms with the corresponding matched entities
    for synonym, entities in synonym_to_entities.items():

      # if cluster consists of 1 entity create that as the candidate entity
        if len(entities) == 1:
            clusters_per_synonym[synonym] = [entities]
            continue

        # Embed  the entities using cosine similarity
        embeddings = model.encode(entities, convert_to_tensor=False)
        similarity = cosine_similarity(embeddings)

        assigned = set()
        clusters = []

        # Loop through the matched entities for a certain synonym
        for i, entity in enumerate(entities):
          
            # Skip entity i if already assigned a cluster
            if i in assigned:
                continue

           # Otherwise add the entity to the specific cluster starting with i
            cluster = [entity]
            assigned.add(i)

           # Loop through the entities that come after i (i+1)
            for j in range(i+1, len(entities)):

               # Check if they are similar enough (>0.9) 
                if j not in assigned and simimilarity[i, j] >= threshold:

                    # If so add them to cluster i
                    cluster.append(entities[j])
                    assigned.add(j)
            clusters.append(cluster)

        # save the clusters
        clusters_per_synonym[synonym] = clusters

    return clusters_per_synonym


def main():
  
  # Load model (SapBERT)
  sapbert = SentenceTransformer('cambridgeltl/SapBERT-from-PubMedBERT-fulltext')
  
  # load cleaned list of entities
  with open("/content/cleaned_unique_entities", "r", encoding="utf-8") as f:
        cleaned_list = json.load(f)
  
  cleaned_set = set(cleaned_list)
  
  # Define the set of matched entities (concept matches and synonyn matches)
  matched_entities = set(exact_matches) | set(synonym_matches)
  
  # Define unmatched entities
  unmatched_entities = list(cleaned_list - matched_entities)
  
  # Embedding for ontology terms
  ontology_embeddings = sapbert.encode(all_ontology_terms, convert_to_tensor=True)
  
  # Create an empty list for the cosine similarity results
  results = []
  
  # Define the cosine similatiy scores and classify unmatched entities
  for entity in unmatched_entities:
  
    # Calculate the embeddings with SapBERT
    embedding = sapbert.encode(entity, convert_to_tensor=True)
  
    # Compute Cosine Similarity score between entities and ontology terms
    cosine_scores = util.pytorch_cos_sim(embedding, ontology_embeddings)[0]
  
    # Obtain the highest result
    top_idx = torch.argmax(cosine_scores).item()
    top_cosine_score = cosine_scores[top_idx].item()
    top_concept_match = all_ontology_terms[top_idx]
  
    if top_cosine_score > 0.9:
      recommend = "Add as synonym"
      relation = "-"
    elif top_cosine_score >= 0.7:
      recommend = "Suggest subclass/related"
      
      # Identify the directionality of the relationship using NLI
      relation = determine_NLI_classes(entity, top_concept_match)
    else:
      recommend = "Manual review"
      relation = "-"
  
    results.append({
      "Extracted entity": entity,
      "Best ontology match": top_concept_match,
      "Cosine similarity score": round(top_cosine_score, 3),
      "Recommendation": recommend,
      "Relationship": relation
    })

    # check for synonym clusters
    synonym_clusters = cluster_synonym(results, sapbert, threshold=0.9)

  

if __name__ == "__main__":
    main()


df = pd.DataFrame(results)
csv_path = "your_path"
df.to_csv(csv_path, index=False)

with open("synonym_clusters_model.json", "w", encoding="utf-8") as f:
    json.dump(synonym_clusters, f, indent=2, ensure_ascii=False)
