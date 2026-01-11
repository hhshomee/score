# :art: SCORE: Specificity, Context Utilization, Robustness, and Relevance for Reference-Free LLM Evaluation 
We propose a multi-dimensional, reference-free evaluation framework that  assesses LLM outputs along four complementary dimensions: specificity, robustness to paraphrasing and semantic perturbations, answer relevance, and context utilization. 

 We introduce a curated dataset of 1,412 domain-specific question–answer pairs spanning 40 professional roles and seven natural hazard types to support systematic evaluation .


<img width="2168" alt="main_fig" src="[https://github.com/hhshomee/score/](https://github.com/hhshomee/score/blob/main/specificity.pdf)">

## Data
:green_book: Questions, Answers, and all the specificity results can be viewed and downloaded [here](https://github.com/hhshomee/score/tree/main/results).


## Specificity
```
python specificity.py --input results/answers/answers.csv --output results/specificity.csv --limit 1500 --llm_type openai--llm_model gpt4o
```

## Answer Relevance
```
python answer_relevance.py --input results/answers/answers.csv --output results/specificity.csv --limit 1500 --llm_type openai--llm_model gpt4o
```

