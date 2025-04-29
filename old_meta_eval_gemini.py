import json
import os
import google.generativeai as genai
from google.generativeai import types

GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    print("Warning: Please set the GOOGLE_API_KEY environment variable.")

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

# Configure the Gemini API
if GOOGLE_API_KEY:
    genai.configure(api_key=GOOGLE_API_KEY)
    # Get the Gemini Pro model
    #model = genai.GenerativeModel('gemini-pro') --> DOESNT WORK
    #model = genai.GenerativeModel('gemini-2.0-flash-thinking-exp')
    model = genai.GenerativeModel('models/gemini-2.0-pro-exp')
else:
    model = None
    print("Gemini Pro model could not be initialized due to missing API key.")

# load data
with open("annotation_data.jsonl") as f:
    data = [json.loads(x) for x in f]

example = data[0]

def evaluate_example(example):
    if model is None:
        print("Skipping evaluation as the Gemini Pro model is not initialized.")
        return None

    print("Model exists...")

    with open("prompts/evaluate.base.txt") as f:
        prompt = f.read().strip()
    human = example["human_summary"]
    system = example["system_summary"]

    # This just formats our prompt with the appropriate human and reference summaries.
    prompt = prompt.replace("{HUMAN}", human).replace("{SYSTEM}", system)

    print("Got the prompt...")
    print(prompt)

    # Get the Gemini response with temperature configuration
    generation_config = genai.types.GenerationConfig(temperature=0.0)
    print("Set the generation config...")
    try:
        print("Let's generate some content...")
        response = model.generate_content([prompt], generation_config=generation_config)

        print("Got the response...")

        # Extract and process the text from the Gemini response
        try:
            gemini_response_text = response.parts[0].text.strip()
            if gemini_response_text.startswith("```") and gemini_response_text.endswith("```"):
                # Remove the backticks and any leading/trailing 'json' if present
                gemini_response_text = gemini_response_text[3:-3].strip().lstrip("json")
            # Assuming the response is a JSON string, load it
            print("About to load the response...")
            response_json = json.loads(gemini_response_text)
            print("llm")
            print(json.dumps(response_json, indent=2))
        except (AttributeError, json.JSONDecodeError) as e:
            print(f"Error processing Gemini response: {e}")
            print(f"Raw Gemini response: {response.text if hasattr(response, 'text') else response}")
            return None

        # system_annotation is how some human did the evaluation.
        print("human")
        system_annotations = example["annotations"][0]["system_annotations"]
        print(json.dumps(system_annotations, indent=2))
        # evaluate whether the LLM evaluation is same as human evaluation
        evaluation_result = dict()
        for attribute in ATTRIBUTE_LIST:
            try:
                llm_prediction = response_json[LLM_ATTRIBUTE_MAP[attribute]]
                llm_prediction = SYS_LABEL_MAP.get(llm_prediction)
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
            except KeyError as e:
                print(f"Warning: Key not found in Gemini response or LLM_ATTRIBUTE_MAP: {e}")
                evaluation_result[attribute] = {
                    "llm_prediction": "not_found",
                    "human_prediction": human_prediction,
                    "correctness": "error",
                }
            except TypeError:
                print(f"Warning: Could not map LLM prediction for attribute: {attribute}, value: {llm_prediction}")
                evaluation_result[attribute] = {
                    "llm_prediction": str(llm_prediction),
                    "human_prediction": human_prediction,
                    "correctness": "error",
                }

        print("evaluation result")
        print(json.dumps(evaluation_result, indent=2))

        return evaluation_result

    except Exception as e:
        print(f"An error occurred during the Gemini API call: {e}")
        return None

# for example in data[:10]:
if model:
    evaluate_example(example)