# CHAIN OF THOUGHT IMPLEMENTATION FOR ANSWER + EXPLANATION (AE) EVAL FOR WIKIWHY
import json

import os
import time
from pathlib import Path

import openai
import pandas as pd
from tqdm import tqdm



 
# define a retry decorator
def retry_with_exponential_backoff(
    func,
    initial_delay: float = 1,
    exponential_base: float = 2,
    jitter: bool = True,
    max_retries: int = 10,
    errors: tuple = (openai.error.RateLimitError,),
):
    """Retry a function with exponential backoff."""
 
    def wrapper(*args, **kwargs):
        # Initialize variables
        num_retries = 0
        delay = initial_delay
 
        # Loop until a successful response or max_retries is hit or an exception is raised
        while True:
            try:
                return func(*args, **kwargs)
 
            # Retry on specific errors
            except errors as e:
                # Increment retries
                num_retries += 1
 
                # Check if max retries has been reached
                if num_retries > max_retries:
                    raise Exception(
                        f"Maximum number of retries ({max_retries}) exceeded."
                    )
 
                # Increment the delay
                delay *= exponential_base * (1 + jitter * random.random())
 
                # Sleep for the delay
                time.sleep(delay)
 
            # Raise exceptions for any errors not specified
            except Exception as e:
                raise e
 
    return wrapper




openai.api_key = os.getenv("OPENAI_API_KEY")

TASK_EXPLANATION = """Generate an answer and a chain of reasoning that leads from the provided question to the answer\n\n"""
TASK_EXAMPLES = """Question: Why did Hiroyuki Yamashita feel pressured writing "Boruto: Naruto the Movie"?
Let's think step by step.
Step 1: Creativity is difficult when put on a strict timetable.
Step 2: There was a need to both produce a good movie and do so on a strict time budget.
Step 3: These two demands put stress on Hiroyuki Yamashita while he worked.
Answer: There were time constraints to writing "Boruto: Naruto the Movie"

Question: Why did Homer P. Rainey get fired by the University of Texas in 1944?
Let's think step by step.
Step 1: The University of Texas was conservative in the 1940s.
Step 2: A conservative institution wouldn't want people working there who have liberal views.
Answer: Homer P. Rainey had liberal views

Question: Why are red maple buds which form in fall and winter often visible from a distance?
Let's think step by step.
Step 1: The color red stands out at a distance.
Step 2: The buds are large, so can be easily seen from far away.
Answer: The large size and reddish tint of red maple buds

Question: Why did the production costs of aluminum change in the late 20th century?
Let's think step by step.
Step 1: Aluminum requires material inputs, energy, and technology to make.
Step 2: In the late 20th century, energy costs continued dropping.
Step 3: Technology to produce aluminum became cheaper and more efficient.
Answer: There were advances in technology, lower energy prices, a favorable exchange rate of the United States dollar, and lower aluminum prices."""

input_csv_path = "../../dataset/v1.1/"
questions = json.load(open(Path(input_csv_path) / "question.json", "r"))["question"]
keys = questions.keys()

split = json.load(open(Path(input_csv_path) / "context.json", "r"))['split']
use_keys = list(filter(lambda x: split[x] == 'dev', keys))

@retry_with_exponential_backoff
def completions_with_backoff(**kwargs):
    return openai.Completion.create(**kwargs)


outputs = {"answer_gpt3" : {}, "cot_gpt3" : {}}

for i in tqdm(range(len(use_keys))):
    print(i)
    question = questions[use_keys[i]]
    response = completions_with_backoff(
        model="text-davinci-003",
        prompt=TASK_EXPLANATION+TASK_EXAMPLES+"Question: " +
        question + "\nLet's think step by step.\n",
        temperature=1,
        max_tokens=1024
    )

    text_out = response["choices"][0]["text"]
    # print(text_out)
    answer_split = text_out.split("\nAnswer: ")
    answer = answer_split[1].strip()
    step_split = answer_split[0].split("\n")
    steps = list(map(lambda x: x.split(":")[-1].strip(), step_split))
    # print(steps)
    # print(answer)
    
    outputs["answer_gpt3"][use_keys[i]] = answer
    outputs["cot_gpt3"][use_keys[i]] = steps

with open("output_gpt3_cot.json","w") as f:
    f.write(json.dumps(outputs))
