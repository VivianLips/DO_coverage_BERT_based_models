# Count the total number of synonyms and concepts in the ontology
number_concepts = len(disease_dict)
number_synonyms = len(synonym_dict)

# Count the total number of concepts and synonyms
ontology_size = number_concepts + number_synonyms

# obtain all entities, exact matches and synonym matches in sets
all_entities = set(cleaned_list)
exact_matches = set(exact_matches)
synonym_matches = set(synonym_matches)
total_matches = exact_matches.union(synonym_matches)

# calculate coverage
exact_match_coverage = len(exact_matches)/len(all_entities)
synonym_match_coverage = len(synonym_matches)/len(all_entities)
total_coverage = len(total_matches)/len(all_entities)

# calculate ontology relevance
ontology_relevance = len(total_matches)/ontology_size

# Print the values
print("Evaluation Metrics:")
print(f"1. Exact Match Coverage:       {exact_match_coverage:.2%}")
print(f"2. Synonym Match Coverage:     {synonym_match_coverage:.2%}")
print(f"3. Total Coverage:             {total_coverage:.2%}")
print(f"5. Ontology Relevance:         {ontology_relevance:.2%}")
