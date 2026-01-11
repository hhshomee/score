import numpy as np
import requests
import openai
from numpy.linalg import norm
import pandas as pd
import json
import re
import os
import time
from tqdm import tqdm
import requests
import argparse
from helpers import (
    get_llm, extract_json_array3,load_dataset)
from sentence_transformers import SentenceTransformer

embedding_model = SentenceTransformer('all-MiniLM-L6-v2')  

def get_embedding(text):
    return embedding_model.encode(text, normalize_embeddings=True)

def cosine_similarity(a, b):
    return np.dot(a, b) / (norm(a) * norm(b) + 1e-10)


def generate_questions(answer, llm_type,llm_model, n=5):
    questions = []
    for _ in range(n):
        prompt = f"Generate a question for the given answer. Only give the question. \nAnswer: {answer}"
        response = get_llm(llm_type, llm_model, prompt)
        questions.append(response.strip())
    return questions

def mask_answer_with_llm(answer, llm_type="argo", llm_model="gpt4o"):
    prompt = f"""
You are a semantic masker. Given the following answer, replace:
- hazard types with [HAZARD]
- profession-related terms with [PROFESSION]
- concern with [CONCERN] (e.g., critical vulnerabilities, maintenance strategies, modernization measures, maintenance strategies, projected impact,design standards, cascading impacts etc.)
- infrastructure with [INFRASTRUCTURE] (e.g., "highway network", "bridge system", "public transit system","railway infrastructure", "airport facilities", "port facilities","freight terminals", "traffic control systems","water treatment plant", "wastewater system", "dam infrastructure", "stormwater system", "coastal protection", "water distribution network","electrical grid", "power distribution network", "EV charging network", "renewable energy infrastructure", "energy storage facilities","power transmission lines", "substations","public buildings", "critical facilities", "commercial structures",etc.)

Keep the structure natural and readable.

Answer:
\"\"\"{answer}\"\"\"
"""
    response = get_llm(llm_type, llm_model, prompt)
    return response.strip()
def compute_answer_relevance(question,gen_question, answer, n=5):
    

    q_emb = get_embedding(question)
    sim_scores = []

    for gen_q in gen_question:
        qi_emb = get_embedding(gen_q)
        sim = cosine_similarity(q_emb, qi_emb)
        sim_scores.append(sim)
    print(sim_scores)
    AR_score = np.mean(sim_scores)
    print(AR_score)
    return AR_score


def evaluate_row(row, idx, llm_type, llm_model):
    print("Procession row number:",idx)
    answer=row["Answer"]
    question=row["question"]
    masked_answer=mask_answer_with_llm(answer)
    gen_question= generate_questions(masked_answer, llm_type,llm_model, n=5)
    score = compute_answer_relevance(question,gen_question, answer, n=5)

    result_dict = {
        "question": question,
        "answer": answer, 
        "maksed_answer":masked_answer,
        "gen_question": gen_question,
        "answer_relevancy_score":score
    }

    return result_dict



def answer_relevancy(input_path, output_path=None, limit=None, llm_type="openai", llm_model="gpt4o"):
    df = load_dataset(input_path, limit)
    df=df[1459:]
    for i, row in tqdm(df.iterrows(), total=len(df)):
        results = []
    file_exists = os.path.exists(output_path) and os.path.getsize(output_path) > 0

    for idx, row in df.iterrows():
        print(f"row number{idx} processing")
        result = evaluate_row(row, idx, llm_type, llm_model)
        results.append(result)
        batch_size=2
        if len(results) >= batch_size:
            batch_df = pd.DataFrame(results)
            batch_df.to_csv(output_path,mode="a" if file_exists else "w", index=False, header=not file_exists)
            file_exists = True
            results = []
            
   
    if results:
        batch_df = pd.DataFrame(results)
        batch_df.to_csv(output_path,mode="a" if file_exists else "w", index=False, header=not file_exists)
       
    
    return pd.read_csv(output_path)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Run robustness scores")
    parser.add_argument("--input", "-i", required=True, help="Input CSV file path")
    parser.add_argument("--output", "-o", help="Output CSV file path")
    parser.add_argument("--limit", "-l", type=int, help="Limit number of rows to process")
    parser.add_argument("--llm_type", type=str, default="argo", help="LLM type: openai, gemini, huggingface")
    parser.add_argument("--llm_model", type=str, default="gpt4o", help="LLM model name")
    
    args = parser.parse_args()   
    
    results = answer_relevancy(args.input, args.output, args.limit, llm_type=args.llm_type,llm_model=args.llm_model)
    print(f"Evaluation complete. Results saved to {args.output or 'default location'}")
#python answer_relevance.py --input results/answer_gemini.csv --output results/answer_relevance.csv --limit 1500 --llm_type openai --llm_model gpt4o
