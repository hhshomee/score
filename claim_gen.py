import csv
import json
import requests
import argparse
from typing import List, Dict, Any, Union, Optional
from ast import literal_eval
import re
def extract_json(text: str) -> List:
    
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


def get_prompt(row):
    

   
   
    prompt = f'''You are an information extractor.
                Your task is to extract only **atomic factual claims** from the answer.
                Extract only atomic factual claims from the input paragraph. Return the output strictly as a JSON array of strings. Do not include explanations.

                Format:
                [
                "Claim 1",
                "Claim 2"
                ]

                Input Paragraph: {row['Answer']}
                '''
    return prompt
    
def fact_extraction(prompt):
    data = {
        "user": "--------",
        "model": "gpt4o",
        "system": "You are a large language model.",
        "prompt": [prompt],
        "stop": [],
        "temperature": 0.1,
        "top_p": 0.9,
        "max_tokens":4096,
        "max_completion_tokens": 1000
    }

    headers = {"Content-Type": "application/json"}
    response = requests.post("----------------",
                             data=json.dumps(data), headers=headers)
    # print(response.json()['response'])
    print(response)
    
    if response.status_code == 200:
        recomm=response.json()['response']
        return extract_json(recomm)
         
    else:
        print(f"Error: {response.status_code}")
        return "ERROR"

def main(input_csv, output_csv):
    with open(input_csv, newline='', encoding='utf-8') as infile, \
         open(output_csv, mode='w', newline='', encoding='utf-8') as outfile:

        reader = csv.DictReader(infile)
        fieldnames = reader.fieldnames + ["Claim"]
        writer = csv.DictWriter(outfile, fieldnames=fieldnames)
        writer.writeheader()
      
        for i, row in enumerate(reader):
            if i>=1860:
                print(f"Generating answer for row {i + 1}...")
                prompt = get_prompt(row)
                answer = fact_extraction(prompt)
                row["Claim"] = answer
                print(answer)
                writer.writerow(row)

    print(f"All answers saved to {output_csv}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate answers using 5 abstracts + question")
    parser.add_argument("--infile", type=str, required=True, help="Path to input CSV")
    parser.add_argument("--outfile", type=str, required=True, help="Path to output CSV")
    args = parser.parse_args()

    main(args.infile, args.outfile)
#python claim_gen.py --infile results/answer.csv  --outfile results/claims.csv