import xml.etree.ElementTree as ET
import numpy as np
import math
from collections import defaultdict
from sklearn.feature_extraction.text import TfidfVectorizer
from rank_bm25 import BM25Okapi

def load_cranfield_dataset(filename):
    """Loads documents from the Cranfield dataset, ensuring correct XML parsing."""
    with open(filename, "r", encoding="utf-8-sig") as file:
        content = file.read().strip()

    # Ensure XML has a single root element
    if not content.startswith("<?xml"):
        content = "<?xml version='1.0' encoding='utf-8'?>\n<root>\n" + content + "\n</root>"

    root = ET.fromstring(content)
    documents = {}

    for doc in root.findall("doc"):
        doc_id_elem = doc.find("docno")
        text_elem = doc.find("text")

        if doc_id_elem is None:
            print("Warning: Skipping document with missing <docno>.")
            continue

        doc_id = doc_id_elem.text.strip()
        text = text_elem.text.strip() if text_elem is not None and text_elem.text else ""  # Handle missing <text>

        documents[doc_id] = text

    return documents

def load_queries(filename):
    """Loads queries from the Cranfield dataset, ensuring correct XML parsing."""
    with open(filename, "r", encoding="utf-8-sig") as file:
        content = file.read().strip()

    if not content.startswith("<?xml"):
        content = "<?xml version='1.0' encoding='utf-8'?>\n<root>\n" + content + "\n</root>"

    root = ET.fromstring(content)
    queries = {}

    for query in root.findall("top"):
        query_id_elem = query.find("num")
        text_elem = query.find("title")

        if query_id_elem is None:
            print("Warning: Skipping query with missing <num>.")
            continue

        query_id = query_id_elem.text.strip()
        text = text_elem.text.strip() if text_elem is not None and text_elem.text else ""  # Handle missing <title>

        queries[query_id] = text

    return queries

def build_tfidf_index(documents):
    """Builds a TF-IDF index using sklearn's TfidfVectorizer."""
    vectorizer = TfidfVectorizer()
    doc_ids = list(documents.keys())
    doc_texts = list(documents.values())
    tfidf_matrix = vectorizer.fit_transform(doc_texts)
    return vectorizer, tfidf_matrix, doc_ids

def build_bm25_index(documents):
    """Builds a BM25 index using the rank_bm25 library."""
    tokenized_corpus = [doc.split() for doc in documents.values()]
    bm25 = BM25Okapi(tokenized_corpus)
    return bm25, list(documents.keys())

def compute_lm_scores(query, documents, lambda_=0.1):
    """Computes scores using a Language Model with Dirichlet smoothing."""
    query_tokens = query.split()
    doc_scores = {}
    collection_freqs = defaultdict(int)
    total_terms = sum(len(doc.split()) for doc in documents.values())

    # Compute collection frequency
    for doc_text in documents.values():
        for token in doc_text.split():
            collection_freqs[token] += 1
    
    for doc_id, text in documents.items():
        doc_tokens = text.split()
        doc_len = len(doc_tokens)
        term_freqs = defaultdict(int)
        
        for token in doc_tokens:
            term_freqs[token] += 1
        
        score = 0
        for token in query_tokens:
            p_td = (term_freqs[token] + 1) / (doc_len + len(collection_freqs))  # Add-one smoothing
            p_tc = collection_freqs[token] / total_terms if collection_freqs[token] > 0 else 1e-10  # Avoid zero division
            score += math.log((lambda_ * p_td) + ((1 - lambda_) * p_tc))
        
        doc_scores[doc_id] = score
    
    return doc_scores

def search_tfidf(query, vectorizer, tfidf_matrix, doc_ids):
    """Searches using TF-IDF and ranks documents."""
    query_vector = vectorizer.transform([query])
    scores = np.dot(tfidf_matrix, query_vector.T).toarray().flatten()
    ranked_docs = sorted(zip(doc_ids, scores), key=lambda x: x[1], reverse=True)
    return ranked_docs

def search_bm25(query, bm25, doc_ids):
    """Searches using BM25 ranking function."""
    scores = bm25.get_scores(query.split())
    ranked_docs = sorted(zip(doc_ids, scores), key=lambda x: x[1], reverse=True)
    return ranked_docs

def write_results(output_file, model_name, results):
    """Writes ranking results to a file in trec_eval format."""
    with open(output_file, "w") as f:
        for query_id, ranked_docs in results.items():
            for rank, (doc_id, score) in enumerate(ranked_docs[:100], start=1):  # Top 100 results
                f.write(f"{query_id} 0 {doc_id} {rank} {score} {model_name}\n")

# Load dataset and queries
documents = load_cranfield_dataset("cran.all.1400.xml")
queries = load_queries("cran.qry.xml")

# Build indexes
vectorizer, tfidf_matrix, doc_ids = build_tfidf_index(documents)
bm25, _ = build_bm25_index(documents)

# Perform retrieval
tfidf_results = {q_id: search_tfidf(q, vectorizer, tfidf_matrix, doc_ids) for q_id, q in queries.items()}
bm25_results = {q_id: search_bm25(q, bm25, doc_ids) for q_id, q in queries.items()}
lm_results = {q_id: sorted(compute_lm_scores(q, documents).items(), key=lambda x: x[1], reverse=True) for q_id, q in queries.items()}

# Write results
write_results("tfidf_results.txt", "TFIDF", tfidf_results)
write_results("bm25_results.txt", "BM25", bm25_results)
write_results("lm_results.txt", "LM", lm_results)

print("Retrieval and ranking complete. Use trec_eval for evaluation.")
