import csv
import argparse
from infrastructure import InfrastructureHazardQuestionGenerator
from utils import load_config
from search import literature_search

class Prompt:
    def __init__(self):
        self.generator = InfrastructureHazardQuestionGenerator()
        self.config = load_config(path="prompt.yml")

    def get_prompt(self, n, file_name):
        questions = self.generator.generate_questions(n)

        
        with open(file_name, mode="w", newline="", encoding="utf-8") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow([
                "Prompt", "literature1", "literature2", "literature3",
                "literature4", "literature5", "context", "question"
            ])

        
        for count, q in enumerate(questions):
            literature, papers = literature_search(q['question'])
            p1, p2, p3, p4, p5 = papers[:5]

            profession_context = self.config["literature_review_instructions"][1]['content'].format(
                profession=q['profession'],
                concern=q['category_type'],
                location=q['location'],
                timeline=q['timeline'],
                scope=q['infrastructure_type'] + " " + q['hazard_type']
            )

            prompt = f"{literature}\n\n{profession_context}\n\n{q['question']}"

           
            with open(file_name, mode="a", newline="", encoding="utf-8") as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow([prompt, p1, p2, p3, p4, p5, profession_context, q['question']])

            print(f"Generated question{count + 1}")

def main(num, output):
    p = Prompt()
    p.get_prompt(num, output)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate Infrastructure-Hazard Questions")
    parser.add_argument("--n", type=int, default=2, help="Number of questions to generate")
    parser.add_argument("--out", type=str, default="results/prompts.csv", help="Output CSV file name")

    args = parser.parse_args()
    main(args.n, args.out)


#python question_gen.py --n 1500 --out results/question.csv
