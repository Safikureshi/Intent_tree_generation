import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from transformers import BertTokenizer, BertModel
import torch
import json
import nltk
from nltk.corpus import stopwords
import psycopg2
import logging

nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

# Initialize BERT model and tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

# Connect to PostgreSQL
conn = psycopg2.connect(
    dbname='Intent_tree',
    user='postgres',
    password='rehan',
    host='localhost',
    port='5432'
)
cur = conn.cursor()


def get_bert_embedding(text):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).squeeze().numpy()


def load_questions(file_path):
    with open(file_path, 'r') as f:
        return [line.strip() for line in f if line.strip()]


def create_initial_clusters(questions):
    return [
        {
            'id': i,
            'parent_id': None,
            'intent_label': None,
            'hierarchical_level': None,
            'questions': [question],
            'centroid': get_bert_embedding(question).tolist(),
            'children': []
        } for i, question in enumerate(questions)
    ]


def merge_clusters(clusters, similarity_threshold, next_id):
    centroids = np.array([cluster['centroid'] for cluster in clusters])
    similarities = cosine_similarity(centroids)
    np.fill_diagonal(similarities, 0)
    merged = set()
    new_clusters = []

    while np.max(similarities) >= similarity_threshold:
        i, j = np.unravel_index(np.argmax(similarities), similarities.shape)
        if i not in merged and j not in merged:
            merged_cluster = {
                'id': next_id,
                'parent_id': None,
                'intent_label': None,
                'hierarchical_level': None,
                'questions': clusters[i]['questions'] + clusters[j]['questions'],
                'centroid': np.mean([clusters[i]['centroid'], clusters[j]['centroid']], axis=0).tolist(),
                'children': [clusters[i]['id'], clusters[j]['id']]
            }
            clusters[i]['parent_id'] = next_id
            clusters[j]['parent_id'] = next_id
            new_clusters.append(merged_cluster)
            merged.add(i)
            merged.add(j)
            next_id += 1

        similarities[i, :] = similarities[:, i] = 0
        similarities[j, :] = similarities[:, j] = 0

    for i, cluster in enumerate(clusters):
        if i not in merged:
            new_cluster = cluster.copy()
            new_cluster['id'] = next_id
            new_cluster['parent_id'] = None
            new_cluster['children'] = [cluster['id']]
            new_clusters.append(new_cluster)
            next_id += 1

    return new_clusters, next_id


def build_hierarchy(initial_clusters, similarity_threshold):
    all_clusters = initial_clusters.copy()
    clusters = initial_clusters
    round = 0
    next_id = len(initial_clusters)

    while (len(clusters) > 1) and (round < 100):  # Adding a limit to prevent infinite loops
        print(f"\nRound: {round}")
        for cluster in clusters:
            print(f"Cluster[{cluster['id']}]: {generate_intent_label(cluster['questions'])}")

        new_clusters, next_id = merge_clusters(clusters, similarity_threshold, next_id)

        if len(new_clusters) == len(clusters):
            similarity_threshold -= 0.01
            print(f"Decreasing similarity threshold to {similarity_threshold}")
            continue

        all_clusters.extend(new_clusters)

        if len(new_clusters) == 1:
            print(f"\nFinal Round: {round + 1}")
            for cluster in new_clusters:
                cluster['hierarchical_level'] = 0
                print(f"Root Cluster[{cluster['id']}]: {generate_intent_label(cluster['questions'])}")
            break

        clusters = new_clusters
        round += 1

    def set_hierarchical_levels(cluster, level):
        cluster['hierarchical_level'] = level
        for child_id in cluster.get('children', []):
            child_cluster = next(c for c in all_clusters if c['id'] == child_id)
            set_hierarchical_levels(child_cluster, level + 1)

    for root in new_clusters:
        set_hierarchical_levels(root, 0)

    return all_clusters, new_clusters


def generate_intent_label(questions):
    vectorizer = TfidfVectorizer(stop_words=list(stop_words), max_features=5)
    X = vectorizer.fit_transform(questions)
    feature_array = np.array(vectorizer.get_feature_names_out())
    tfidf_sorting = np.argsort(X.toarray()).flatten()[::-1]
    top_n = feature_array[tfidf_sorting][:5]
    return ', '.join(top_n)


def print_cluster_tree(clusters, roots):
    def print_node(node_id, level=0):
        indent = "  " * level
        node = next(cluster for cluster in clusters if cluster['id'] == node_id)
        print(
            f"{indent}Cluster[{node_id}] (Level: {node['hierarchical_level']}): {generate_intent_label(node['questions'])}")
        for child_id in node.get('children', []):
            print_node(child_id, level + 1)

    for root in roots:
        print(
            f"Root Cluster[{root['id']}] (Level: {root['hierarchical_level']}): {generate_intent_label(root['questions'])}")
        for child_id in root.get('children', []):
            print_node(child_id, 1)


def bfs_search(cluster_id, target_embedding, all_clusters, similarity_threshold):
    queue = [cluster_id]
    matched_clusters = []

    while queue:
        current_id = queue.pop(0)
        current_cluster = next(c for c in all_clusters if c['id'] == current_id)
        current_similarity = cosine_similarity([current_cluster['centroid']], [target_embedding])[0][0]

        if current_similarity >= similarity_threshold:
            matched_clusters.append(current_cluster['id'])

        queue.extend(current_cluster['children'])

    return matched_clusters


def insert_question(question, all_clusters, similarity_threshold):
    question_embedding = get_bert_embedding(question)
    root_clusters = [cluster['id'] for cluster in all_clusters if cluster['parent_id'] is None]
    matched_clusters = []

    for root_id in root_clusters:
        matched_clusters.extend(bfs_search(root_id, question_embedding, all_clusters, similarity_threshold))

    if matched_clusters:
        matched_clusters = matched_clusters[:5]  # Only consider the top 5 matched clusters
        for cluster_id in matched_clusters:
            matched_cluster = next(cluster for cluster in all_clusters if cluster['id'] == cluster_id)
            matched_cluster['questions'].append(question)
            new_centroid = np.mean([matched_cluster['centroid'], question_embedding], axis=0)
            matched_cluster['centroid'] = new_centroid.tolist()
            try:
                cur.execute(
                    "UPDATE intent_embeddings SET embedding = %s WHERE treeid = %s",
                    (json.dumps(new_centroid.tolist()), matched_cluster['id'])
                )
                logging.debug(f"Updated centroid for cluster {matched_cluster['id']}")
            except Exception as e:
                logging.error(f"Error updating centroid for cluster {matched_cluster['id']}: {e}")
        conn.commit()
        print(f"Added question to existing Clusters: {matched_clusters}")
    else:
        new_cluster = {
            'id': len(all_clusters),
            'parent_id': None,
            'intent_label': None,
            'hierarchical_level': None,
            'questions': [question],
            'centroid': question_embedding.tolist(),
            'children': []
        }
        all_clusters.append(new_cluster)
        try:
            cur.execute(
                "INSERT INTO intent_embeddings (treeid, embedding) VALUES (%s, %s)",
                (new_cluster['id'], json.dumps(new_cluster['centroid']))
            )
            logging.debug(f"Inserted new centroid for cluster {new_cluster['id']}")
        except Exception as e:
            logging.error(f"Error inserting centroid for new cluster {new_cluster['id']}: {e}")
        conn.commit()
        print(f"Created new Cluster[{new_cluster['id']}]")

    return matched_clusters, all_clusters


def store_intent_tree(questions, tree_structure):
    query = """
    INSERT INTO intent_tree (questions, tree)
    VALUES (%s, %s)
    RETURNING treeid;
    """
    logging.debug("Storing intent tree...")
    try:
        cur.execute(query, (questions, json.dumps(tree_structure)))
        treeid = cur.fetchone()[0]
        conn.commit()
        logging.debug(f"Stored intent tree with treeid: {treeid}")
        return treeid
    except Exception as e:
        logging.error(f"Error storing intent tree: {e}")
        conn.rollback()


def update_intent_tree(treeid, new_question):
    query = """UPDATE intent_tree SET questions = array_cat(questions, ARRAY[%s]) WHERE treeid = %s"""
    try:
        # Check if treeid exists in the intent_embeddings table
        cur.execute("SELECT nodeid FROM intent_embeddings WHERE treeid = %s", (treeid,))
        result = cur.fetchone()
        if not result:
            logging.error(f"No entry found for treeid: {treeid} in intent_embeddings")
            return None

        cluster_id = result[0]

        # Update the intent tree with the new question
        cur.execute(query, (new_question, treeid))
        insert_question(new_question, cluster_id, similarity_threshold=0.7)
        conn.commit()
        logging.debug(f"Updated intent tree with treeid: {treeid}")
    except Exception as e:
        logging.error(f"Error updating intent tree: {e}")
        conn.rollback()


def store_embeddings(treeid, embeddings):
    query = """
    INSERT INTO intent_embeddings (treeid, embedding)
    VALUES (%s, %s)
    """
    logging.debug(f"Storing embeddings for treeid: {treeid}")
    try:
        for embedding in embeddings:
            cur.execute(query, (treeid, json.dumps(embedding)))
        conn.commit()
        logging.debug(f"Stored embeddings for treeid: {treeid}")
    except Exception as e:
        logging.error(f"Error storing embedding: {e}")
        conn.rollback()

def construct_intent_tree(questions, similarity_threshold):
    initial_clusters = create_initial_clusters(questions)
    all_clusters, root_clusters = build_hierarchy(initial_clusters, similarity_threshold)

    for cluster in all_clusters:
        cluster['intent_label'] = generate_intent_label(cluster['questions'])

    print_cluster_tree(all_clusters, root_clusters)

    tree_structure = {cluster['id']: cluster for cluster in all_clusters}
    treeid = store_intent_tree(questions, tree_structure)

    # Save the hierarchy to a JSON file
    with open('intent_hierarchy.json', 'w') as f:
        json.dump(tree_structure, f, indent=2)
    embedding = [cluster['centroid'] for cluster in all_clusters]
    store_embeddings(treeid, embedding)

    return {"success": "OK", "intentTreeID": treeid}

questions = load_questions('Intent_Question1.txt')
result = construct_intent_tree(questions, similarity_threshold=0.7)
print(result)

def main():
    tree_id = int(input("Enter the tree ID: "))
    new_questions = input("Enter the questions (comma-separated): ")
    new_questions = [q.strip() for q in questions]
    #similarity_threshold = float(input("Enter similarity threshold (0-1): "))
    result = update_intent_tree(tree_id, new_questions)
    print(result)

if __name__ == "__main__":
    main()
