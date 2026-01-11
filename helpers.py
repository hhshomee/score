import json
import re
import os
import time
import logging
from typing import List, Dict, Any, Union, Optional
import csv
import pandas as pd
from langchain_community.chat_models import ChatOpenAI
from langchain_community.embeddings import OpenAIEmbeddings
from sentence_transformers import SentenceTransformer
import google.generativeai as genai
import requests
from transformers import BitsAndBytesConfig
# from config import OPENAI_API_KEY, DEFAULT_LLM_MODEL,DEFAULT_LLM_TYPE
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch
from ast import literal_eval

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def get_qwen(model_name="Qwen/Qwen3-8B", device="cuda"):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map="auto"
    )

    def qwen_invoke(messages, temperature=0.1, max_new_tokens=4096):
        prompt = messages[0].content if messages else ""
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature
        )
        text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        return type("Response", (), {"content": text})()

    return type("LLMWrapper", (), {
        "invoke": staticmethod(qwen_invoke)
    })()

def get_llm(llm_type, model_name,prompt):

    if llm_type == "openai":

        # Make sure you have: export OPENAI_API_KEY="..."
       
        return ChatOpenAI(
            model_name=model_name,   
            max_tokens=4096,
        )

  
    elif llm_type == "llama":
      
        def ollama_llm_invoke(messages,model_name=model_name, temperature=0.1):
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
                logger.error(f"Ollama LLM failed: {e}")
                raise

        return type("LLMWrapper", (), {
            "invoke": staticmethod(ollama_llm_invoke)
        })()
    elif llm_type == "qwen":
        return get_qwen(model_name)
    
def get_embeddings():
   
    return SentenceTransformer("all-MiniLM-L6-v2")



def extract_json_array(text: str) -> List:
    
    try:
        parsed = json.loads(text)
        if isinstance(parsed, list):
            return parsed
    except json.JSONDecodeError:
        pass

   
    for match in re.finditer(r"\[.*?\]", text, re.DOTALL):
        snippet = match.group(0)

       
        try:
            parsed = json.loads(snippet)
            if isinstance(parsed, list):
                return parsed
        except Exception:
            pass

        
        try:
            parsed = json.loads(snippet.replace("'", '"'))
            if isinstance(parsed, list):
                return parsed
        except Exception:
            pass

       
        try:
            parsed = literal_eval(snippet)
            if isinstance(parsed, list):
                return parsed
        except Exception:
            pass

    logger.warning("Failed to extract JSON array. Raw text: %s", text)
    return []
def extract_json_array2(text: str) -> List:
    import json
    import re

    try:
        parsed = json.loads(text)
        if isinstance(parsed, list):
            return parsed
    except json.JSONDecodeError:
        pass

    try:
        matches = re.findall(r"\[\s*(?:\"?(?:true|false)\"?\s*,\s*)*(?:\"?(?:true|false)\"?)\s*\]", text.lower())
        for m in matches:
            try:
                parsed = json.loads(m.replace("'", '"'))
                if isinstance(parsed, list):
                    return parsed
            except:
                continue
    except Exception as e:
        pass

    logger.warning("Failed to extract JSON array. Raw text: %s", text[:500])
    return []
import json
import re

def extract_json_array3(response):
    text = response.content if hasattr(response, "content") else str(response)

  
    text = re.sub(r"```(?:json)?", "", text).strip("` \n")

 
    text = re.sub(r'(\])\s*("claim\d+":)', r'\1,\n\2', text)  # if array precedes claim key
    text = re.sub(r'("\]")\s*("claim\d+":)', r'\1,\n\2', text)
    text = re.sub(r'("\}")\s*("claim\d+":)', r'\1,\n\2', text)
    text = re.sub(r'("})\s*("claim\d+":)', r'\1,\n\2', text)
    text = re.sub(r'("])\s*("claim\d+":)', r'\1,\n\2', text)

  
    text = re.sub(r'("\])\s*(")', r'\1,\n\2', text)
    text = re.sub(r'("\})\s*(")', r'\1,\n\2', text)

 
    text = re.sub(r'("\])\s*(")', r'\1,\n\2', text)

    try:
        return json.loads(text)
    except json.JSONDecodeError as e:
        print("Still failed JSON parse:", e)
        return {}


def extract_float(text: str) -> float:
    
    try:
        score = float(text.strip())
        return max(0.0, min(score, 1.0))
    except ValueError:
       
        match = re.search(r'(\d+\.\d+|\d+)', text)
        if match:
            try:
                score = float(match.group(1))
               
                if score > 1:
                    score = score / 100 if score <= 100 else 0.9
                return max(0.0, min(score, 1.0))
            except:
                pass
        
        logger.warning("Could not extract float from: %s", text[:100])
        return 0.5

def load_dataset(file_path: str, limit: Optional[int] = None) -> pd.DataFrame:
    
    try:
        df = pd.read_csv(file_path)
        logger.info(f"Loaded {len(df)} rows from {file_path}")
        
        if limit is not None and limit > 0:
            df = df.head(limit)
            logger.info(f"Limited to {limit} rows")
        
        return df
    except Exception as e:
        logger.error(f"Error loading dataset: {e}")
        raise

def save_results(results: pd.DataFrame, output_path: str):
    
    try:
        results.to_csv(output_path, index=False)
        logger.info(f"Saved results to {output_path}")
        return True
    except Exception as e:
        logger.error(f"Error saving results: {e}")
        return False
