import csv
import json
import requests
import argparse

def get_prompt(row,question,context,lit1,lit2,lit3,lit4,lit5):

        return f"""You are tasked with writing a recommendation/fact-based answer that answers the user’s question based on a provided list of research abstracts and contextual information. Your response must:

                1. Directly address the user's concern, ensuring the answer is supported by the provided literature.
                2. Incorporate the user's profile like timeline, professional background, Location, and concern  into the recommendation.
                3. Clearly connect insights from the abstracts to the user's specific context and goals.
                4. make sure to ouput in points (1,2,3..) without inserting any **.
                5. End your response with a confidence score (in percentage) and a short explanation for that score.

                Here are the 5 research abstracts:
                1. {lit1}
                2. {lit2}
                3. {lit3}
                4. {lit4}
                5. {lit5}

                Context: {context}

                Question: {question}

                Based on the above abstracts write a the answer in points.  Make sure to take into account all the information in the context like profession, timeline, etc. Do not include subpoints.
                """
   

def gen_ans(prompt):
    data = {
        "user": "------",
        "model": "gpt4o",
        "system": "You are a large language model.",
        "prompt": [prompt],
        "stop": [],
        "temperature": 0.1,
        "top_p": 0.9,
        "max_tokens":4096,
        "max_completion_tokens": 1000,
        "logprobs":True, 
    }

    headers = {"Content-Type": "application/json"}
    response = requests.post("---------------",
                             data=json.dumps(data), headers=headers)
    # print(response.json()['response'])
    print(response)
    
    if response.status_code == 200:
        return response.json()['response']
    else:
        print(f"Error: {response.status_code}")
        return "ERROR"

def main(input_csv, output_csv):
    with open(input_csv, newline='', encoding='utf-8') as infile, \
         open(output_csv, mode='w', newline='', encoding='utf-8') as outfile:

        reader = csv.DictReader(infile)
        fieldnames = reader.fieldnames + ["Answer"]
        writer = csv.DictWriter(outfile, fieldnames=fieldnames)
        writer.writeheader()

        for i, row in enumerate(reader):
            print(f"Generating answer for row {i + 1}...")
            prompt = get_prompt(row)
            answer = gen_ans(prompt)
            row["Answer"] = answer
            print(answer)
            writer.writerow(row)

    print(f"All answers saved to {output_csv}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate answers using 5 abstracts + question")
    parser.add_argument("--infile", type=str, required=True, help="Path to input CSV")
    parser.add_argument("--outfile", type=str, required=True, help="Path to output CSV")
    args = parser.parse_args()

    main(args.infile, args.outfile)
#python ans_gen.py --infile results/question.csv --outfile results/answer.csv 