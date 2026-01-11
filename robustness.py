import os
import re
import json
import os
import ast
import time
import math
import logging
import numpy as np
import pandas as pd
import argparse
from typing import List, Dict, Any, Optional
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langchain_core.messages import HumanMessage
import pandas as pd
from sentence_transformers import SentenceTransformer, util
from ans_gen import get_prompt,gen_ans 
from langchain_core.messages import HumanMessage
from helpers import (
    get_llm, extract_json_array3,load_dataset)

from search import literature_search
def get_llm2( model_name='llama3.1', temperature=0.7):
     
            def ollama_llm_invoke(messages,model_name='llama3.1', temperature=0.7):
                prompt = messages[0].content if messages else ""
                import requests 
                try:
                    response = requests.post(
                        "http://localhost:11434/api/generate",
                        json={
                            "model": model_name,     # e.g., "llama3"
                            "prompt": prompt,
                            "stream": False,
                            "options": {
                                "temperature": temperature,
                                "num_predict": 512
                            }
                        }
                    )
                    response.raise_for_status()
                    return type("Response", (), {"content": response.json()["response"]})()
                except Exception as e:
                   pass

            return type("LLMWrapper", (), {
                "invoke": staticmethod(ollama_llm_invoke)
            })()
model = SentenceTransformer("all-MiniLM-L6-v2")

def semantic_similarity(text1, text2):
    """Compute cosine similarity between two texts using SBERT."""
    emb1 = model.encode(text1, convert_to_tensor=True)
    emb2 = model.encode(text2, convert_to_tensor=True)
    return float(util.cos_sim(emb1, emb2)[0][0])
def para(question,llm_type, llm_model):
    
    prompt = f'''You are a query paraphrase specialist.
              Create a variation of the original question by SUBSTITUTING VOCABULARY.
              Replace keywords with synonyms, use more formal/informal language.
              Preserve the core meaning and information need.
              Output ONLY the perturbed question with no explanations.
              Original question: {question}'''

    if llm_type =='ollama':

        llm = get_llm2(llm_model,prompt)
        msg = llm.invoke([HumanMessage(content=prompt)])
        return msg.content
    elif llm_type=='argo':
         
         msg = get_llm(llm_type,llm_model,prompt)
       
         return msg

def perturbed(question,llm_type, llm_model):
    prompt = f'''You are a query perturber specialist.
              Create a variation of the original question by SUBSTITUTING hazard types, and location
              Output ONLY the perturbed question with no explanations.
              Original question: {question}'''
    if llm_type =='ollama':
        llm = get_llm2(llm_model,prompt)
        msg = llm.invoke([HumanMessage(content=prompt)])
        return msg.content
    elif llm_type=='argo':
         msg = get_llm(llm_type,llm_model,prompt)
         return msg
def perturbed2(context,llm_type, llm_model):
    prompt = f'''You are a user profile perturber specialist.
              Create a variation of the original profile by SUBSTITUTING profession.
              The list of profession you can choose from:
              profession = [
                            # TRANSPORTATION
                            "Highway Engineer", "Bridge Inspector", "Railway Systems Engineer",
                            "Transit Operations Manager", "Airport Infrastructure Manager",
                            "Port Facility Manager", "Transportation Safety Inspector",
                            "Traffic Systems Engineer", "Pavement Engineer", "Transportation Planner",

                            # WATER
                            "Water Systems Engineer", "Hydraulic Engineer", "Dam Safety Inspector",
                            "Wastewater Treatment Specialist", "Maritime Infrastructure Manager",
                            "Stormwater Engineer", "Water Quality Specialist",
                            "Coastal Infrastructure Engineer",

                            # ENERGY
                            "Power Systems Engineer", "Electrical Grid Manager",
                            "Energy Distribution Specialist", "EV Infrastructure Planner",
                            "Renewable Energy Systems Manager", "Transmission Line Engineer",
                            "Substation Engineer", "Energy Storage Specialist",

                            # BUILDINGS
                            "Structural Engineer", "Building Systems Manager", "Facilities Manager",
                            "Real Estate Asset Manager", "Building Automation Specialist",
                            "Construction Manager", "Building Code Inspector", "MEP Systems Engineer",

                            # COMMUNICATIONS
                            "Telecommunications Engineer", "Broadband Infrastructure Specialist",
                            "Network Resilience Manager", "Data Center Infrastructure Engineer",
                            "Fiber Optics Specialist", "Communications Systems Planner",
                            "Network Security Engineer"
                        ]

              Output ONLY the perturbed profile with no explanations.
              Original question: {context}'''
    if llm_type =='ollama':
        llm = get_llm2(llm_model,prompt)
        msg = llm.invoke([HumanMessage(content=prompt)])
        return msg.content
    elif llm_type=='argo':
         msg = get_llm(llm_type,llm_model,prompt)
        
         return msg
def generate_answer(llm_type,row,paraphrased_question,context,llm_model,p1= None,p2=None,p3=None,p4=None,p5=None):
    if p1==None:
        lit1=row['literature1']
        lit2=row['literature2']
        lit3=row['literature3']
        lit4=row['literature4']
        lit5=row['literature5']
        # context=row['context']
        
        prompt = get_prompt(row,paraphrased_question,context,lit1,lit2,lit3,lit4,lit5)
        # print("Prompt for para",prompt)
        if llm_type =='ollama':
            llm = get_llm2(llm_model)
            msg = llm.invoke([HumanMessage(content=prompt)])
            return msg.content
        elif llm_type=='argo':
            msg = get_llm(llm_type,llm_model,prompt)
            return msg

    else:
        prompt = get_prompt(row,paraphrased_question,context,p1,p2,p3,p4,p5)
      

        
        # if llm_type =='ollama':
        #     llm = get_llm2(llm_model)
        #     msg = llm.invoke([HumanMessage(content=prompt)])
        #     return msg.content
        # elif llm_type=='argo':
        msg = get_llm(llm_type,llm_model,prompt)
        return msg


def evaluate_row(row, idx, llm_type, llm_model):
    
    question=row["question"]
    answer=row["Answer"]
    context=row['context']
   
    paraphrased_question=para(question,llm_type, llm_model)
    paraphrased_answer=generate_answer(llm_type,row,paraphrased_question,context, llm_model)
    para_sim=semantic_similarity(answer,paraphrased_answer)
    
    #for changing hazard type, location
    perturbed_question=perturbed(row["question"],llm_type, llm_model)
    literature, papers = literature_search(perturbed_question)
    p1, p2, p3, p4, p5 = papers[:5]
    perturbed_answer=generate_answer(llm_type,row,perturbed_question,context,llm_model, p1, p2, p3, p4, p5) 
    pert_sim=semantic_similarity(answer,perturbed_answer)
    
    #for changing profile
    perturbed_context=perturbed2(row["context"],llm_type, llm_model)
    perturbed_answer2=generate_answer(llm_type,row,question,perturbed_context,llm_model)
    pert_sim2=semantic_similarity(answer,perturbed_answer2)
    
    print(question)
    print(perturbed_question)
    print(context)
    print(perturbed_context)

    
    
    

   
    

    

    result_dict = {
            "question": question,
            "answer": answer,
            "context":context,
            "paraphrased_question": paraphrased_question,
            "paraphrased_answer": paraphrased_answer,
            "perturbed_question": perturbed_question, #for changing hazard type
            "perturbed_answer": perturbed_answer,
            "perturbed_context": perturbed_context, #for changing profession
            "perturbed_context_answer": perturbed_answer2,
            "literaaature1_per":p1,
            "literaaature2_per":p2,
            "literaaature3_per":p3,
            "literaaature4_per":p4,
            "literaaature5_per":p5,
            "para_sim": para_sim,
            "pert_sim": pert_sim,
            "pert_context_sim": pert_sim2

    }
    return result_dict




def robustness(input_path, output_path=None, limit=None, llm_type="openai", llm_model="gpt4o"):
    df = load_dataset(input_path, limit)
    df=df[1155:]
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
            #logger.info(f"Saved batch of {batch_size} rows to {output_path}")
   
    if results:
        batch_df = pd.DataFrame(results)
        batch_df.to_csv(output_path,mode="a" if file_exists else "w", index=False, header=not file_exists)
        #logger.info(f"Saved final batch of {len(results)} rows to {output_path}")
    
    return pd.read_csv(output_path)



if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Run robustness scores")
    parser.add_argument("--input", "-i", required=True, help="Input CSV file path")
    parser.add_argument("--output", "-o", help="Output CSV file path")
    parser.add_argument("--limit", "-l", type=int, help="Limit number of rows to process")
    parser.add_argument("--llm_type", type=str, default="openai", help="LLM type: openai, gemini, huggingface")
    parser.add_argument("--llm_model", type=str, default="gpt4o", help="LLM model name")
    
    args = parser.parse_args()
   
    
    results = robustness(args.input, args.output, args.limit, llm_type=args.llm_type,llm_model=args.llm_model)
    print(f"Evaluation complete. Results saved to {args.output or 'default location'}")



#python robustness.py --input results/answer.csv --output results/robustness.csv --limit 1500 --llm_type openai--llm_model gpt4o
