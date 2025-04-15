from llm import GPT
import json
import os

# Variables
# Read the API key and organization ID from the file
with open("keys/openai.key") as f:
    KEY = f.read().strip()

with open("keys/openai.id") as f:
    ORG = f.read().strip()

ATTRIBUTE_LIST = ["plaintiff", "defendant", "filing_date", "court_name", "statutory", "holding", "outcome", "judge", "outcome_date"]

REF_LABEL_MAP = {
    1: "included",
    2: "not_included",
}

SYS_LABEL_MAP = {
    1: "included_complete",
    2: "included_contradiction",
    3: "included_extra",
    4: "included_incomplete",
    5: "not_included",
}

LLM_ATTRIBUTE_MAP = {
    "plaintiff": "Plaintiff",
    "defendant": "Defendant",
    "filing_date": "Initial filing date",
    "court_name": "Court's full name",
    "statutory": "Issue or statutory basis for the case",
    "holding": "Holding",
    "outcome": "Remedial outcome",
    "judge": "Judge's Name",
    "outcome_date": "Date of holding or outcome",
}  # this should be changed according to the prompt



# Get the model
model = GPT(
    model_pt="gpt-4o-mini",
    key=KEY,
    account=ORG,
    parallel_size=1
)

# load data

with open("annotation_data.jsonl") as f:
    data = [json.loads(x) for x in f]

example = data[0]

def evaluate_example(example):
    with open("prompts/evaluate.base.txt") as f:
        prompt = f.read().strip()
    human = example["human_summary"]
    system = example["system_summary"]

    prompt = prompt.replace("{HUMAN}", human).replace("{SYSTEM}", system)

    # print(prompt)

    # print(model.get_response([{"role": "user", "content": prompt}], temperature=0.0))
    response = model.get_response([{"role": "user", "content": prompt}], temperature=0.0)

    response = response[0]["text"].split("\n", 1)[1].rsplit("\n", 1)[0]
    response = json.loads(response)
    print("llm")
    print(json.dumps(response, indent=2))
    print("human")
    system_annotations = example["annotations"][0]["system_annotations"]
    print(json.dumps(system_annotations, indent=2))
    # evaluation
    evaluation_result = dict()
    for attribute in ATTRIBUTE_LIST:
        llm_prediction = response[LLM_ATTRIBUTE_MAP[attribute]]
        llm_prediction = SYS_LABEL_MAP[llm_prediction]
        human_prediction = system_annotations[attribute]["label"]
        if llm_prediction == human_prediction:
            correctness = "correct"
        else:
            correctness = "incorrect"
        evaluation_result[attribute] = {
            "llm_prediction": llm_prediction,
            "human_prediction": human_prediction,
            "correctness": correctness,
        }
    print("evaluation result")
    print(json.dumps(evaluation_result, indent=2))
        
    return evaluation_result

# for example in data[:10]:
evaluate_example(example)