The intent tree update still in progress...little glitch is there. It will done within this week.
The description of each function in the Intent Tree Builder Module:

1. load_questions(file_path)

- Loads questions from a text file, where each line represents a question.
- Returns a list of questions.

2. create_initial_clusters(questions)

- Creates initial clusters based on the BERT embeddings of the input questions.
- Returns a list of initial clusters, where each cluster contains a single question.

3. build_hierarchy(initial_clusters, similarity_threshold)

- Builds a hierarchical clustering structure by merging clusters based on their similarity.
- Uses cosine similarity to determine the similarity between clusters.
- Returns the updated list of clusters and the root clusters.

4. generate_intent_label(questions)  [Optional]

- Generates an intent label for a list of questions using TF-IDF vectorization.
- Returns the intent label.

5. store_intent_tree(questions, tree_structure)

- Stores the intent tree structure in a PostgreSQL database.
- Returns the tree ID.

6. update_intent_tree(tree_id, new_questions, all_clusters, similarity_threshold)

- Updates the intent tree structure with new questions.
- Recalculates the cluster structure based on the new questions.
- Returns the updated list of clusters.

7. construct_intent_tree(questions, similarity_threshold)

- Builds the intent tree structure from scratch.
- Calls create_initial_clusters, build_hierarchy, and store_intent_tree functions.
- Returns the tree ID.

8. print_cluster_tree(clusters, roots)

- Prints the cluster tree structure in a readable format.

9. bfs_search(cluster_id, target_embedding, all_clusters, similarity_threshold)

- Performs a breadth-first search to find clusters similar to the target embedding.
- Returns a list of matched cluster IDs.

10. insert_question(question, cluster_ids, all_clusters, similarity_threshold)

- Inserts a new question into the intent tree structure.
- Updates the cluster structure based on the new question.
- Returns the updated list of clusters.
