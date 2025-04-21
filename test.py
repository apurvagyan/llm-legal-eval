from meta_eval import MetaEvaluator
from llm import GPT

# Initialize model with your API key
with open("keys/openai.key") as f:
    KEY = f.read().strip()

# The organization ID might be empty for personal accounts
try:
    with open("keys/openai.id") as f:
        ORG = f.read().strip()
except:
    ORG = None

# Initialize the model
model = GPT(
    model_pt="gpt-4o-mini",
    key=KEY,
    account=ORG,
    parallel_size=1
)

# Test with a single example
evaluator = MetaEvaluator()
example = evaluator.data[0]
result = evaluator.evaluate_example(example, model)
print(result)
