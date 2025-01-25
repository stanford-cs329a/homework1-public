import os
import json
import subprocess
from datasets import Dataset, DatasetDict
import random

TEST_FILE_URL = "https://github.com/openai/prm800k.git"
TEST_FILE_NAME = "test.jsonl"
TRAIN_FILE_NAME = "train.jsonl"
# https://github.com/ScalingIntelligence/large_language_monkeys/blob/ed2bdb06bcfcbda1ed7d9c2bff87c2db10c3ee78/llmonk/generate/prompts.py#L5

MATH_COT_PROMPT = """Problem:
Find the domain of the expression  $\\frac{\\sqrt{x-2}}{\\sqrt{5-x}}$.}

Solution:
The expressions inside each square root must be non-negative. Therefore, $x-2 \\ge 0$, so $x\\ge2$, and $5 - x \\ge 0$, so $x \\le 5$. Also, the denominator cannot be equal to zero, so $5-x>0$, which gives $x<5$. Therefore, the domain of the expression is $\\boxed{[2,5)}$.\nFinal Answer: The final answer is $[2,5)$

Problem:
If $\\det \\mathbf{A} = 2$ and $\\det \\mathbf{B} = 12,$ then find $\\det (\\mathbf{A} \\mathbf{B}).$

Solution:
We have that $\\det (\\mathbf{A} \\mathbf{B}) = (\\det \\mathbf{A})(\\det \\mathbf{B}) = (2)(12) = \\boxed{24}.$\nFinal Answer: The final answer is $24$

Problem:
Terrell usually lifts two 20-pound weights 12 times. If he uses two 15-pound weights instead, how many times must Terrell lift them in order to lift the same total weight?

Solution:
If Terrell lifts two 20-pound weights 12 times, he lifts a total of $2\\cdot 12\\cdot20=480$ pounds of weight.  If he lifts two 15-pound weights instead for $n$ times, he will lift a total of $2\\cdot15\\cdot n=30n$ pounds of weight.  Equating this to 480 pounds, we can solve for $n$:\n\\begin{align*}\n30n&=480\\\n\\Rightarrow\\qquad n&=480/30=\\boxed{16}\n\\end{align*}\nFinal Answer: The final answer is $16$

Problem:
If the system of equations\n\n\\begin{align*}\n6x-4y&=a,\\\n6y-9x &=b.\n\\end{align*}has a solution $(x, y)$ where $x$ and $y$ are both nonzero,\nfind $\\frac{a}{b},$ assuming $b$ is nonzero.

Solution:
If we multiply the first equation by $-\\frac{3}{2}$, we obtain\n\n$$6y-9x=-\\frac{3}{2}a.$$Since we also know that $6y-9x=b$, we have\n\n$$-\\frac{3}{2}a=b\\Rightarrow\\frac{a}{b}=\\boxed{-\\frac{2}{3}}.$$\nFinal Answer: The final answer is $-\\frac{2}{3}$"""


def load_hendrycks_math(root_dir: str, split: str = "test") -> Dataset:
    file_name = TEST_FILE_NAME if split == "test" else TRAIN_FILE_NAME
    file_path = os.path.join(root_dir, file_name)

    # Download the file if it doesn't exist using git lfs
    if not os.path.exists(file_path):
        subprocess.run(["git", "lfs", "clone", TEST_FILE_URL], cwd=root_dir, check=True)
        # Move the downloaded file to the desired location
        subprocess.run(
            [
                "mv",
                os.path.join(
                    root_dir, "prm800k/prm800k/math_splits/{}.jsonl".format(split)
                ),
                file_path,
            ],
            check=True,
        )
        # Clean up the cloned repository
        subprocess.run(["rm", "-rf", os.path.join(root_dir, "prm800k")], check=True)

    # Read and process the file
    processed_data = []
    with open(file_path, "r") as f:
        for line in f:
            item = json.loads(line)
            processed_item = {
                "problem": item["problem"],
                # "solution": item["solution"],
                "answer": item["answer"],
            }
            processed_data.append(processed_item)
    return Dataset.from_list(processed_data)


class MATH500:
    """
    A class for the MATH500 dataset.
    The original math dataset is available at https://github.com/hendrycks/math.
    We are using the MATH500 split, because a lot of the later works to train reward models, such as the [OpenAI PRM800k](https://github.com/openai/prm800k), use this dataset.
    """

    def __init__(self, root_dir: str = "./data", split: str = "test"):
        self.root_dir = root_dir
        os.makedirs(root_dir, exist_ok=True)
        self.dataset = load_hendrycks_math(root_dir, split)
        self.rng = random.Random(42)

    def get_problems(self, debug_mode: bool = False) -> list[dict]:
        """
        Returns a list of all problems in the dataset.
        Each problem is a dictionary containing 'problem', 'solution', and 'answer' keys.
        Problems are randomly shuffled with a fixed seed for reproducibility.
        """
        problems = self.dataset.to_list()
        self.rng.shuffle(problems)
        
        if debug_mode:
            return problems[:10]
        else:
            # By default, we'll run on at most 50 problems.
            return problems[:50]

    def get_system_prompt(self) -> str:
        return (
            "You are a language model that solves math problems. Think step by step. Use the below format when responding."
            + "\n"
            + MATH_COT_PROMPT
        )
