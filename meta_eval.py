import json
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter, defaultdict
from typing import Dict, List, Tuple, Any, Optional
import importlib
import google.generativeai as genai
from llm import GPT
import time

def list_gemini_models():
    models = genai.list_models()
    for model in models:
        print(f"Model: {model.name}")
        for method in model.supported_generation_methods:
            print(f"- Supports: {method}")

PROMPT = "prompts/evaluate.base.txt"

MODEL = {"provider": "gemini", "model_name": "gemini-2.0-flash"}  # "gemini" or "chatgpt"
# MODEL = {"provider": "chatgpt", "model_name": "gpt-4o-mini"}
DELAY = 5

# Constants
ATTRIBUTE_LIST = ["plaintiff", "defendant", "filing_date", "court_name", "statutory",
                  "holding", "outcome", "judge", "outcome_date"]

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
}

SYS_LABEL_MAP = {
    1: "included_complete",
    2: "included_contradiction",
    3: "included_extra",
    4: "included_incomplete",
    5: "not_included",
}

REF_LABEL_MAP = {
    1: "included",
    2: "not_included",
}

# Invert maps for easy lookup
INV_SYS_LABEL_MAP = {v: k for k, v in SYS_LABEL_MAP.items()}
INV_REF_LABEL_MAP = {v: k for k, v in REF_LABEL_MAP.items()}
INV_LLM_ATTRIBUTE_MAP = {v: k for k, v in LLM_ATTRIBUTE_MAP.items()}

class MetaEvaluator:
    """
    Evaluates LLM-generated annotations against human annotations
    and calculates performance metrics.
    """

    def __init__(self, data_path: str = "annotation_data.jsonl"):
        """
        Initialize the evaluator with the path to the annotation data.

        Args:
            data_path: Path to the JSONL file containing annotations
        """
        self.data_path = data_path
        self.data = self._load_data()
        self.evaluation_results = []
        self.metrics = {}
    def _load_data(self) -> List[Dict]:
        """Load annotation data from JSONL file."""
        with open(self.data_path, 'r') as f:
            return [json.loads(line) for line in f if line.strip()]

    def evaluate_all_examples(self, llm_model, save_results: bool = True) -> Dict:
        """
        Evaluate all examples in the dataset.

        Args:
            llm_model: LLM model instance for evaluation
            save_results: Whether to save results to file

        Returns:
            Dictionary of evaluation metrics
        """
        results = []
        for i, example in enumerate(self.data):
            print(f"Evaluating example {i+1}/{len(self.data)}")
            result = self.evaluate_example(example, llm_model)
            time.sleep(DELAY) # so I don't hit the quota lol
            results.append(result)

        self.evaluation_results = results
        self.metrics = self.calculate_metrics(results)

        if save_results:
            with open("evaluation_results.json", "w") as f:
                json.dump({
                    "results": results,
                    "metrics": self.metrics
                }, f, indent=2)

        return self.metrics

    def evaluate_example(self, example: Dict, llm_model) -> Dict:
        """
        Evaluate a single example.

        Args:
            example: Annotation example data
            llm_model: LLM model instance for evaluation

        Returns:
            Dict containing evaluation results
        """
        # Get human and system summaries
        human_summary = example["human_summary"]
        system_summary = example["system_summary"]

        # Prepare prompt
        with open(PROMPT) as f:
            prompt = f.read().strip()

        prompt = prompt.replace("{HUMAN_SUMMARY}", human_summary).replace("{SYSTEM_SUMMARY}", system_summary)

        # Get model response
        if MODEL["provider"] == "gemini":
            response = llm_model.get_response([prompt], temperature=0.0)  # Call the wrapper's method
            response_text = response[0]["text"]
        elif MODEL["provider"] == "chatgpt":
            response = llm_model.get_response([{"role": "user", "content": prompt}], temperature=0.0)
            response_text = response[0]["text"]
        else:
            raise ValueError(f"Invalid model provider: {MODEL['provider']}")

        # Extract JSON from response
        try:
            # Find the JSON part (usually between ```json and ```)
            json_start = response_text.find('```json')
            if json_start != -1:
                json_text = response_text[json_start + 7:]
                json_end = json_text.find('```')
                if json_end != -1:
                    json_text = json_text[:json_end].strip()
            else:
                # Try to find any json-like structure
                import re
                json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
                if json_match:
                    json_text = json_match.group(0)
                else:
                    # Fallback to the entire response
                    json_text = response_text

            llm_evaluation = json.loads(json_text)

        except json.JSONDecodeError as e:
            print(f"Error parsing JSON from LLM response: {e}")
            print(f"Response was: {response_text}")
            # Create dummy response with all 5s (not included)
            llm_evaluation = {attr: 5 for attr in LLM_ATTRIBUTE_MAP.values()}

        # Get human annotations
        system_annotations = example["annotations"][0]["system_annotations"]

        # Create evaluation result
        evaluation_result = {
            "example_id": example.get("example_id", "unknown"),
            "doc_id": example.get("doc_id", "unknown"),
            "attributes": {}
        }

        # Compare LLM evaluation with human annotations
        for attribute in ATTRIBUTE_LIST:
            llm_key = LLM_ATTRIBUTE_MAP[attribute]

            # Get LLM prediction (numeric value 1-5)
            llm_prediction_numeric = llm_evaluation.get(llm_key, 5)  # Default to 5 (not included) if missing
            llm_prediction = SYS_LABEL_MAP[llm_prediction_numeric]

            # Get human annotation
            human_prediction = system_annotations[attribute]["label"]

            # Determine correctness
            correctness = "correct" if llm_prediction == human_prediction else "incorrect"

            # Add to results
            evaluation_result["attributes"][attribute] = {
                "llm_prediction": llm_prediction,
                "llm_prediction_numeric": llm_prediction_numeric,
                "human_prediction": human_prediction,
                "correctness": correctness
            }

        return evaluation_result

    def calculate_metrics(self, results: List[Dict]) -> Dict:
        """
        Calculate evaluation metrics from results.

        Args:
            results: List of evaluation results

        Returns:
            Dict containing metrics
        """
        # Initialize counters
        total = len(results)
        correct_by_attribute = defaultdict(int)
        label_distribution = defaultdict(lambda: defaultdict(int))
        confusion_matrix = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))
        # Process results
        for result in results:
            for attribute, data in result["attributes"].items():
                # Count correct predictions
                if data["correctness"] == "correct":
                    correct_by_attribute[attribute] += 1
                # Count label distribution
                human_label = data["human_prediction"]
                llm_label = data["llm_prediction"]
                label_distribution[attribute][human_label] += 1

                # Build confusion matrix
                confusion_matrix[attribute][human_label][llm_label] += 1

        # Calculate accuracy metrics
        accuracy_by_attribute = {
            attribute: count / total for attribute, count in correct_by_attribute.items()
        }

        overall_accuracy = sum(correct_by_attribute.values()) / (total * len(ATTRIBUTE_LIST))

        # Create detailed metrics
        metrics = {
            "overall_accuracy": overall_accuracy,
            "accuracy_by_attribute": accuracy_by_attribute,
            "label_distribution": dict(label_distribution),
            "confusion_matrix": {
                attr: {
                    human: dict(llm_counts) for human, llm_counts in attr_data.items()
                } for attr, attr_data in confusion_matrix.items()
            },
            "total_examples": total
        }

        return metrics

    def visualize_results(self, save_path: str = "evaluation_visualizations"):
        """
        Generate visualizations of evaluation results.

        Args:
            save_path: Directory to save visualizations
        """
        if not self.metrics:
            print("No metrics available. Run evaluate_all_examples first.")
            return

        os.makedirs(save_path, exist_ok=True)

        # 1. Accuracy by attribute
        accuracy = self.metrics["accuracy_by_attribute"]
        plt.figure(figsize=(12, 6))
        plt.bar(accuracy.keys(), accuracy.values())
        plt.axhline(self.metrics["overall_accuracy"], color='r', linestyle='--',
                    label=f'Overall Accuracy: {self.metrics["overall_accuracy"]:.2f}')
        plt.title("Accuracy by Attribute")
        plt.ylabel("Accuracy")
        plt.xticks(rotation=45, ha="right")
        plt.ylim(0, 1)
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"{save_path}/accuracy_by_attribute.png")
        # 2. Confusion matrices for each attribute
        for attribute in ATTRIBUTE_LIST:
            # Get attribute confusion matrix
            conf_matrix = self.metrics["confusion_matrix"][attribute]
            # Get all possible labels that appear in the data
            all_labels = set()
            for human_label in conf_matrix.keys():
                all_labels.add(human_label)
                all_labels.update(conf_matrix[human_label].keys())

            all_labels = sorted(list(all_labels))

            # Create confusion matrix array
            matrix_array = np.zeros((len(all_labels), len(all_labels)))

            # Map from label to index
            label_to_idx = {label: i for i, label in enumerate(all_labels)}

            # Fill matrix
            for human_label, llm_predictions in conf_matrix.items():
                for llm_label, count in llm_predictions.items():
                    human_idx = label_to_idx[human_label]
                    llm_idx = label_to_idx[llm_label]
                    matrix_array[human_idx, llm_idx] = count
            # Plot confusion matrix
            plt.figure(figsize=(10, 8))
            plt.imshow(matrix_array, interpolation='nearest', cmap=plt.cm.Blues)
            plt.title(f"Confusion Matrix: {attribute}")
            plt.colorbar()
            tick_marks = np.arange(len(all_labels))
            plt.xticks(tick_marks, all_labels, rotation=45, ha="right")
            plt.yticks(tick_marks, all_labels)
            plt.xlabel("LLM Prediction")
            plt.ylabel("Human Annotation")
            # Add text annotations
            thresh = matrix_array.max() / 2
            for i in range(len(all_labels)):
                for j in range(len(all_labels)):
                    plt.text(j, i, format(int(matrix_array[i, j]), 'd'),
                             ha="center", va="center",
                             color="white" if matrix_array[i, j] > thresh else "black")

            plt.tight_layout()
            plt.savefig(f"{save_path}/confusion_matrix_{attribute}.png")

        # 3. Label distribution
        for attribute in ATTRIBUTE_LIST:
            distribution = self.metrics["label_distribution"][attribute]
            labels = list(distribution.keys())
            counts = list(distribution.values())
            plt.figure(figsize=(10, 6))
            plt.bar(labels, counts)
            plt.title(f"Label Distribution: {attribute}")
            plt.ylabel("Count")
            plt.tight_layout()
            plt.savefig(f"{save_path}/label_distribution_{attribute}.png")
        print(f"Visualizations saved to {save_path}")

    def generate_report(self, save_path: str = "evaluation_report.md"):
        """
        Generate a detailed Markdown report of evaluation results.
        Args:
            save_path: Path to save the report
        """
        if not self.metrics:
            print("No metrics available. Run evaluate_all_examples first.")
            return
        report = [
            "# Legal Summary Evaluation Report\n",
            f"Total examples evaluated: {self.metrics['total_examples']}\n",
            f"Overall accuracy: {self.metrics['overall_accuracy']:.4f}\n",
            "\n## Accuracy by Attribute\n",
        ]
        # Add accuracy by attribute table
        report.append("| Attribute | Accuracy |\n")
        report.append("| --- | --- |\n")
        for attribute, accuracy in self.metrics["accuracy_by_attribute"].items():
            report.append(f"| {attribute} | {accuracy:.4f} |\n")
        # Add label distribution
        report.append("\n## Label Distribution\n")
        for attribute in ATTRIBUTE_LIST:
            report.append(f"\n### {attribute}\n")
            report.append("| Label | Count |\n")
            report.append("| --- | --- |\n")

            distribution = self.metrics["label_distribution"][attribute]
            for label, count in distribution.items():
                report.append(f"| {label} | {count} |\n")

        # Add confusion matrices
        report.append("\n## Confusion Matrices\n")
        for attribute in ATTRIBUTE_LIST:
            report.append(f"\n### {attribute}\n")
            conf_matrix = self.metrics["confusion_matrix"][attribute]
            # Get all possible labels
            all_labels = set()
            for human_label in conf_matrix.keys():
                all_labels.add(human_label)
                all_labels.update(conf_matrix[human_label].keys())

            all_labels = sorted(list(all_labels))

            # Create table header
            report.append("| Human ↓ / LLM → | " + " | ".join(all_labels) + " |\n")
            report.append("| --- | " + " | ".join(["---"] * len(all_labels)) + " |\n")

            # Fill table
            for human_label in all_labels:
                row = [f"| {human_label}"]
                for llm_label in all_labels:
                    count = conf_matrix.get(human_label, {}).get(llm_label, 0)
                    row.append(f" {count}")
                row.append(" |\n")
                report.append(" |".join(row))

        # Write report
        with open(save_path, "w") as f:
            f.writelines(report)

        print(f"Report saved to {save_path}")

# Main execution when run as script
if __name__ == "__main__":
    if MODEL["provider"] == "chatgpt":

        # Initialize model (ChatGPT)
        with open("keys/openai.key") as f:
            KEY = f.read().strip()
        with open("keys/openai.id") as f:
            ORG = f.read().strip()

        model = GPT(
            model_pt=MODEL["model_name"],
            key=KEY,
            account=ORG,
            parallel_size=1
        )

    elif MODEL["provider"] == "gemini":
        # Initialize Gemini
        GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")
        if not GOOGLE_API_KEY:
            raise ValueError("GOOGLE_API_KEY environment variable not set.")
        genai.configure(api_key=GOOGLE_API_KEY)
        gemini_model = genai.GenerativeModel(MODEL["model_name"])  # e.g., "gemini-pro"

        # Define a wrapper to match the expected interface
        class GeminiWrapper:
            def __init__(self, gemini_model):
                self.gemini_model = gemini_model

            def get_response(self, prompt_list, temperature=0.0):
                # Adapt Gemini's input format
                contents = [{"parts": [{"text": prompt}]} for prompt in prompt_list]
                response = self.gemini_model.generate_content(contents,
                                                            generation_config=genai.types.GenerationConfig(
                                                                temperature=temperature))
                # Adapt Gemini's response format to match the 'llm' library
                return [{"role": "assistant", "text": response.text}]

        model = GeminiWrapper(gemini_model)

    else:
        raise ValueError(f"Invalid model provider: {MODEL['provider']}")

    # Create evaluator
    evaluator = MetaEvaluator()

    # Run evaluation
    metrics = evaluator.evaluate_all_examples(model)

    # Generate visualizations and report
    evaluator.visualize_results()
    evaluator.generate_report()
