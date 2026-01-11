import sys
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

from helpers import (
    get_llm, extract_json_array3,load_dataset)
   
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
logger = logging.getLogger(__name__)


class SpecState(TypedDict):
    question: str
    abs_claims: list
    claims: list
    llm_type: str
    llm_model: str
    knowledge: str
    specific_info: str
    result_1: list
    result_2: list
    result_3: list
    result_1_msg: Any
    result_2_msg: Any
    result_3_msg: Any
    result_1_parsed: dict
    result_2_parsed: dict
    result_3_parsed: dict
    msg_1: str
    msg_2: str
    msg_3: str
    parsed_1: dict
    parsed_2: dict
    parsed_3: dict
    final_score: float
    claim_scores: list
    dimension_averages: list
    rounded_averages: list
   
def spec_extraction(state: SpecState):
    llm_type = state["llm_type"]
    llm_model = state["llm_model"]

    prompt = f'''You are a  reasoning agent that extracts specific information from claims, and literature abstracts.

                Your task is to:
                **Extract Specific Information**: Identify specific facts or entities from  claims, and literature abstracts. These may include the following.

                Hazard types: wildfire, heat weave, storm, etc.
                Location: County and States like Cook County, Arizona, etc.
                Timeline: 2018, In past decade, 5-10 years, etc.
                Intensity: increasing, extreme cold, etc.

                Example 
                - From Claims: [high temperature, vulnerability, Arizona,midwest, wildfire, 5-10 years] 
                

                Input:
    
                Claim: {state['claims']}
                '''
    
    
    llm = get_llm(llm_type, llm_model, prompt)
    
    msg = llm.invoke([HumanMessage(content=prompt)])
    # msg = get_llm(llm_type,llm_model,prompt)
    # print("from spec extraction",msg)
    # print("SPEC",msg.content)
 
    # print("========Specificity==========")
    return {"specific_info": msg} 
def get_agent_scores(parsed):

    claim_scores = []
    claim_numbers = set()
    for key in parsed.keys():
        if 'claim' in key and key.replace('claim', '').isdigit():
            claim_numbers.add(int(key.replace('claim', '')))
    
    claim_numbers = sorted(claim_numbers)
    
    # For each claim, get [hazard, location, timeline, intensity]
    for claim_num in claim_numbers:
        hazard = parsed.get(f'hazard{claim_num}', 'n/a').strip().lower()
        location = parsed.get(f'location{claim_num}', 'n/a').strip().lower()
        timeline = parsed.get(f'timeline{claim_num}', 'n/a').strip().lower()
        intensity = parsed.get(f'intensity{claim_num}', 'n/a').strip().lower()
        
        # Convert to numbers: yes=1, no=0, n/a stays as 'n/a'
        def convert_score(score):
            if score == 'yes':
                return 1
            elif score == 'no':
                return 0
            else:
                return 'n/a'
        
        claim_score = [convert_score(hazard), convert_score(location), 
                      convert_score(timeline), convert_score(intensity)]
        claim_scores.append(claim_score)
    
    return claim_scores
def normalize_msg_dict(msg):
    merged = {}
    for i, raw in msg.items():
        if not raw or not isinstance(raw, str):
            
            continue
        raw = raw.strip().strip("'").strip("`")

        
        match = re.search(r"\{.*\}", raw, re.S)
        if not match:
            print(f"Claim {i}: no JSON object found in string")
            continue

        candidate = match.group(0)

       
        try:
            parsed = json.loads(candidate)
        except Exception as e:
          
            continue
        
        # try:
        #     parsed = json.loads(raw)   # parse JSON string to dict
        # except Exception as e:
        #     print(f"Error parsing claim {i}: {e}")
        #     continue

        for k, v in parsed.items():
           
            if re.search(r"\d+$", k):
                merged[k] = v
            else:
                merged[f"{k}{i}"] = v
    return merged
def agent_generic(state: SpecState, label: str):
   
    llm_type = state["llm_type"]
    llm_model = state["llm_model"]
    claims=state['claims']
    msg_dict= {}
    for i, claim in enumerate(claims, start=1):
        # print(i)

        prompt =f''' You are a strict evaluator of **specificity**.

                    For each case, you are given:
                    - A factual **claim**
                    - A list of **evidence passages** from a trusted source
                    - A set of **specific details** extracted from the claim (hazard type, location, timeline, intensity)

                    Your task is to evaluate the claim using ONLY the provided evidence.
                    ----------------------------
                    LABEL DEFINITIONS (IMPORTANT)
                    ----------------------------

                    For each specific detail (hazard, location, timeline, intensity), use EXACTLY one of the following labels:

                    - "yes":
                    The detail is explicitly mentioned in the claim AND
                    it matches the same specific detail discussed in the knowledge source.

                    - "no":
                    The detail is explicitly mentioned in the claim BUT
                    the knowledge source does NOT provide sufficient information to verify it.
                    (This includes cases where the evidence contradicts the claim or does not confirm it.)

                    - "N/A":
                    The detail is NOT mentioned in the claim at all.

                    For location matching, agreement at the STATE level is sufficient; an exact county or city match is not required.
                    Do NOT infer or assume any facts beyond the evidence.
                    Lack of verification MUST be labeled as "no" (not "N/A").

                    ----------------------------
                    EVALUATION STEPS
                    ----------------------------


                    Your task is to:
                    1. Determine whether the **claim is factually true, false, or partially correct**, using ONLY the evidence.
                    2. For each of the 4 specific details (hazard, location, timeline, intensity):
                     - Assign "yes", "no", or "N/A" based on the rules above.
                     - Provide a brief factual explanation for your decision.
                    
                    3. Justify your overall factuality decision concisely and objectively.
                    4. If the claim is "true", cite the exact evidence passage(s) that support it.
                    5. If the claim is "false" or "partially correct", explain precisely which details are unsupported or incorrect.

                    ----------------------------
                    OUTPUT FORMAT
                    ----------------------------

                    Return your answer as a SINGLE JSON object in the following format
                    (with no markdown, no extra text, and no explanations outside the JSON):
                
                    {{
                    "claim": "<Claim>",
                    
                    "hazard": "yes" | "no" | "N/A",
                    "hazard_reasoning": "<Explain whether hazard mentioned in the claim is explicitly supported>",
                    
                    "location": "yes" | "no" | "N/A",
                    "location_reasoning": "<Explain whether location mentioned in the claim is supported>",
                    
                    "timeline": "yes" | "no" | "N/A",
                    "timeline_reasoning": "<Explain whether timeline like date and range of years mentioned in the claim is supported>",
                    
                    "intensity": "yes" | "no" | "N/A",
                    "intensity_reasoning": "<Explain whether intensity  mentioned in the claim is supported>"
                    
                    }}

                    ----------------------------
                    INPUTS
                    ----------------------------


                    Claim: {claim}

                    Specific Details to Check: {state['specific_info']}

                    Evidence Passages: {state['knowledge']}

                '''
        
        
        llm = get_llm(llm_type, llm_model, prompt)
        msg = llm.invoke([HumanMessage(content=prompt)])
        msg_dict[i]=msg.content
        
    parsed = normalize_msg_dict(msg_dict)

    try:
        agent_scores = get_agent_scores(parsed)
       
    except Exception as e:
        #logger.error(f"Error getting scores for agent {label}: {e}")
        agent_scores = []
    
    return {
        label: agent_scores,
        f"{label}_msg": msg_dict,
        f"{label}_parsed": parsed
    }

def agent_1(state: SpecState): return agent_generic(state, "result_1")
def agent_2(state: SpecState): return agent_generic(state, "result_2")
def agent_3(state:SpecState): return agent_generic(state, "result_3")




def vote_majority(agent1_score, agent2_score, agent3_score):
    """
    Vote among three options: n/a, 0, and 1
    Return whichever has the majority/plurality
    """
    scores = [agent1_score, agent2_score, agent3_score]
    
    
    na_votes = scores.count('n/a')
    no_votes = scores.count(0)
    yes_votes = scores.count(1)
    
     # In case of ties, prioritize: n/a > 0 > 1 
    if na_votes > no_votes and na_votes > yes_votes:
        return 'n/a'
    elif no_votes > yes_votes:
        return 0
    elif yes_votes > no_votes:
        return 1
    else:
       
        if na_votes == no_votes and na_votes > yes_votes:
            return 'n/a'  # Tie between n/a and 0, choose n/a
        elif na_votes == yes_votes and na_votes > no_votes:
            return 'n/a'  # Tie between n/a and 1, choose n/a
        elif no_votes == yes_votes and no_votes > na_votes:
            return 0  # Tie between 0 and 1, choose 0 
        else:
            # Three-way tie, default to n/a 
            return 'n/a'

def get_agreed_scores(agent1_scores, agent2_scores, agent3_scores):
   
    all_scores = [agent1_scores, agent2_scores, agent3_scores]
    if not any(scores for scores in all_scores):
        # print("All agent scores are empty!")
        return []
    max_claims = max(len(scores) if scores else 0 for scores in all_scores)
    
    claim_scores = []
    
    for idx in range(max_claims):
        
        claim_consensus = []
        
        # Get this claim's scores from each agent
        claim1 = agent1_scores[idx] if agent1_scores and idx < len(agent1_scores) else ['n/a', 'n/a', 'n/a', 'n/a']
        claim2 = agent2_scores[idx] if agent2_scores and idx < len(agent2_scores) else ['n/a', 'n/a', 'n/a', 'n/a']
        claim3 = agent3_scores[idx] if agent3_scores and idx < len(agent3_scores) else ['n/a', 'n/a', 'n/a', 'n/a']
        
        
        # Vote on each dimension
        dimensions = ['hazard', 'location', 'timeline', 'intensity']
        for dim_idx in range(4):
            score1 = claim1[dim_idx] if dim_idx < len(claim1) else 'n/a'
            score2 = claim2[dim_idx] if dim_idx < len(claim2) else 'n/a'
            score3 = claim3[dim_idx] if dim_idx < len(claim3) else 'n/a'
            
            consensus_vote = vote_majority(score1, score2, score3)
            claim_consensus.append(consensus_vote)
            
            # print(f"  {dimensions[dim_idx]}: [{score1}, {score2}, {score3}] → {consensus_vote}")
        
        claim_scores.append(claim_consensus)
        # print(f"Claim {idx + 1} consensus: {claim_consensus}")
    
    
    return claim_scores

def calculate_final_score(claim_scores):
    
    weights = [0.6, 0.2, 0.1, 0.1]  # hazard, location, timeline, intensity
    dimensions = ['hazard', 'location', 'timeline', 'intensity']
    
    dimension_scores = [[], [], [], []]
    
    for claim in claim_scores:
       
        for dim_idx in range(4):
            if dim_idx < len(claim) and claim[dim_idx] != 'n/a':
                dimension_scores[dim_idx].append(claim[dim_idx])
                
    # Calculate averages
    dimension_averages = []
    for scores in dimension_scores:
        if scores:
            avg = sum(scores) / len(scores)
            dimension_averages.append(avg)
            
        else:
            dimension_averages.append('n/a')
    rounded_averages = []
    for avg in dimension_averages:
        if isinstance(avg, (int, float)):
            rounded_averages.append(1.0 if avg >= 0.5 else 0.0)
        else:
            rounded_averages.append('n/a')
      
   
    weighted_sum = 0.0
    valid_weight_sum = 0.0
    for i in range(4):
        avg = dimension_averages[i]
        if isinstance(avg, (int, float)):
            weighted_sum += avg * weights[i]
            valid_weight_sum += weights[i]
    final_score = weighted_sum / valid_weight_sum if valid_weight_sum > 0 else 0.0

    return final_score, dimension_averages, rounded_averages



def aggregator(state: SpecState):
    
    claims = state.get("claims", [])
    
    
    agent1_scores = state.get("result_1", [])
    agent2_scores = state.get("result_2", [])
    agent3_scores = state.get("result_3", [])
    
    msgs = [state.get(f"result_{i+1}_msg", "") for i in range(3)]
    parseds = [state.get(f"result_{i+1}_parsed", "") for i in range(3)]
    

    # print(f"\n=== AGGREGATOR START ===")
    
    try:
       
        claim_scores = get_agreed_scores(agent1_scores, agent2_scores, agent3_scores)
        
        final_score, dimension_averages, rounded_averages = calculate_final_score(claim_scores)
        
        
    except Exception as e:
        logger.error(f"Error in aggregator: {e}")
   
    result = {
        "final_score": final_score,
        "claim_scores": claim_scores,
        "dimension_averages": dimension_averages,
        "rounded_averages": rounded_averages,
        "msg_1": msgs[0], 
        "msg_2": msgs[1], 
        "msg_3": msgs[2],
        "parsed_1": parseds[0], 
        "parsed_2": parseds[1], 
        "parsed_3": parseds[2]
    }
    
    print(f"=== AGGREGATOR END ===\n")
    return result

def create_graph():
    workflow = StateGraph(SpecState)
    
    # nodes
    workflow.add_node("spec_extraction", spec_extraction)
    workflow.add_node("agent_1", agent_1)
    workflow.add_node("agent_2", agent_2)
    workflow.add_node("agent_3", agent_3)
    workflow.add_node("aggregator", aggregator)
    
    # edges
    workflow.add_edge(START, "spec_extraction")
    for agent in ["agent_1", "agent_2", "agent_3"]:
        workflow.add_edge("spec_extraction", agent)
        workflow.add_edge(agent, "aggregator")
    workflow.add_edge("aggregator", END)
    
    return workflow.compile()

def evaluate_row(graph, row, idx, llm_type, llm_model):

    print(f"row number {idx} processing")
    
    question=row["question"]
    # abs_claims=ast.literal_eval(row['AbsClaim'])
    claims=ast.literal_eval(row['Claim'])
   
    knowledge_text = " ".join(str(row[f"literature{i}"]) for i in range(1, 6) if f"literature{i}" in row)
    
    start_time = time.time()
    try:
        # state = graph.invoke({"question": question, "abs_claims": abs_claims,"claims":claims,"llm_type": llm_type,"llm_model": llm_model,"knowledge":knowledge_text})
        state = graph.invoke({"question": question,"claims":claims,"llm_type": llm_type,"llm_model": llm_model,"knowledge":knowledge_text})

        #logger.info(f"Finished row {idx} in {round(time.time() - start_time, 2)}s")
        
        final_score = state.get("final_score", 0.0)
        msg_1=state.get("msg_1", "")
        # print(msg_1,"msgs")
        
        result_dict = {
            "question": state['question'],
            "knowledge": state["knowledge"],
            "claims": state["claims"],
            "agent1_scores": json.dumps(state.get("result_1", [])),
            "agent2_scores": json.dumps(state.get("result_2", [])),
            "agent3_scores": json.dumps(state.get("result_3", [])),
            "claim_scores": json.dumps(state.get("claim_scores", [])),
            "dimension_averages": json.dumps(state.get("dimension_averages", [])),
            "rounded_averages": state['rounded_averages'],
            "final_score": state["final_score"],
            "specificity": state['specific_info'],
            "msg_1": state.get("msg_1", ""),
            "msg_2": state.get("msg_2", ""),
            "msg_3": state.get("msg_3", ""),
            "parsed_1": state.get("parsed_1", ""),
            "parsed_2": state.get("parsed_2", ""),
            "parsed_3": state.get("parsed_3", ""),

        }
        
        return result_dict
        
    except Exception as e:
        #logger.error(f"Error on row {idx}: {e}")
        raise

def specificity(input_path, output_path=None, limit=None, llm_type="argo", llm_model="gpt4o"):
    
    df = load_dataset(input_path, limit)
    # df=df[897:890]
    graph = create_graph()
    
    results = []
    file_exists = os.path.exists(output_path) and os.path.getsize(output_path) > 0

    for idx, row in df.iterrows():
        result = evaluate_row(graph, row, idx, llm_type, llm_model)
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

    parser = argparse.ArgumentParser(description="Run specificity scoring")
    parser.add_argument("--input", "-i", required=True, help="Input CSV file path")
    parser.add_argument("--output", "-o", help="Output CSV file path")
    parser.add_argument("--limit", "-l", type=int, help="Limit number of rows to process")
    parser.add_argument("--llm_type", type=str, default="argo", help="LLM type: openai, gemini, huggingface")
    parser.add_argument("--llm_model", type=str, default="gpt4o", help="LLM model name")
    
    args = parser.parse_args()
    
    results = specificity(args.input, args.output, args.limit, llm_type=args.llm_type,llm_model=args.llm_model)
    print(f"Evaluation complete. Results saved to {args.output or 'default location'}")


#python specificity.py --input results/answer.csv --output results/specificity.csv --limit 1500 --llm_type openai--llm_model gpt4o