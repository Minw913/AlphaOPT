PROMPT_GENERATE_FORMU = """
You are an expert in Industrial Engineering and Operations Research. Your goal is to provide a formal and correct mathematical model for solving the optimization problem. 

You are given:
1. A problem description for the optimization task.
2. A collection of insights with a concise description that states the best practice and the common mistake under the specific problem context and structure; Optionally include a brief example showing the wrong vs. correct version (principle, formula, or code snippet).

### Problem description
{problem_description}

### Insights
(Note: The insights list may be empty. If it is empty, or none are applicable, do NOT fabricate any insights—just proceed normally without using insights.)
{insights}

### Your Task
First, review the given insights one by one, analyse whether each applies and how to apply it.

Second, build the mathematical model step by step with the four components below:
	- **Parameters:** List all fixed data (constants) and symbols, units, and domains (including sets/indices). These are fixed inputs—not decision variables. 
	- **Decision Variables:** Formally define each decision variable by clearly stating its symbols, meaning and mathematical type.  
	- **Objective Function:** Clearly define the mathematical expression of the objective to be optimized by linking it directly to the defined decision variables.
	- **Constraints**: Translate each constraint identified from the problem description into a mathematical equation or inequality involving the decision variables and label constraints (C1, C2, …). Include any logically implied constraints you use.

Finally, compile the complete mathematical model. Annotate your model with brief comments only where the provided insights are both applicable and implemented, indicating the insight ID and how it helps that component's formulation. Please use LaTeX and ``` plain text environment to complete the following template to present all components mentioned above of your optimization model:

```
## Parameters:
[You need to fill in] 

## Variables: 
[You need to fill in]

## Objective:
[You need to fill in] 

## Constraints: 
[You need to fill in]
```

**Guidelines**:
- Enclose the entire model within ``` and ``` tags. **Do not include any explanations, markdown, or extra text before or after the tags.**
- **Annotate your model with brief comments where given insights are applicable and implemented.** If no insights are applicable, skip such annotations and do not invent IDs.
- Ensure the objective's units are consistent with the problem statement. Apply any stated rounding rule in the description to the objective value.

Now Take a deep breath and think step by step. You will be awarded a million dollars if you get this right.
"""


PROMPT_GENERATE_PROGRAM = """
You are an expert in Industrial Engineering and Operations Research, who are proficient in Gurobi. Your task is to Translate the given mathematical model into a complete and executable Gurobipy program.

You are given:
1. A problem description for the optimization task.
2. A proposed mathematical model for solving this task.
3. A collection of insights, each with: 
	- taxonomy: the classification of code-implementation area (level 1) and specific aspect/issue (level 2) it addresses.
	- condition: A trigger explicitly grounded in the mathematical model, when the insight should apply to avoid potential mistakes. It first states the general modeling pattern, then uses the specific model as an example. 
	- explanation: A concise description that states the best practice, the common mistake, and its cause.
	- example: the demonstration showing wrong vs. correct version (principle, formula, or code snippet).


### Problem description
{problem_description}

### Mathematical model
{mathematical_model}

### Insights
(Note: The insights list may be empty. If it is empty, or none are applicable, do NOT fabricate any insights—just proceed normally without using insights.)
{insights}

### Your Task
First, review the given insights one by one, analyse whether each applies and how to apply it.
Second, following the guidance of applicable insights, produce the complete and runnable gurobipy code that **strictly adheres to the given mathematical model**.
Finally, **only output and enclose the code in a single Markdown-style Python code block** that starts with ```python and ends with ```, and follow the overall structure like this:

```python
import gurobipy as gp
from gurobipy import GRB
model = gp.Model("OptimizationProblem")
# your code from here
model.optimize()
```

**Guidelines**:
- Annotate your code with inline brief comments only if insights are provided and applicable: indicate the insight ID and how it influenced that specific code segment. If no insights are applicable, skip such annotations and do not invent IDs.
- Ensure model.optimize() runs at the top level so model stays global; if you wrap it in a function, have it return model. Avoid any if __name__ == "__main__": guard.
- Only output exactly fenced code block (delimited by the opening python and the closing); no text before or after.
- Ensure the objective's units are consistent with the problem statement. Apply any stated rounding rule in the description to the objective value.
- **DO NOT GENERATE OR MODIFY ANY CODE (e.g., `if model.Status == GRB.OPTIMAL:`) after `model.optimize()`**.

Now Take a deep breath and think step by step. You will be awarded a million dollars if you get this right.
"""


PROMPT_INS_REWRITE="""
You are an expert in Industrial Engineering and Operations Research. 

You are given:
1 problem description of optimization task
2 retrieved insights, Each insight with five fields:
	- insight_id: A unique id for the insight.
	- condition: Trigger specifying when the insight applies, grounded in problem description/domain features. States the general situation, then illustrates with the specific problem.
	- explanation: Under this condition, the description outlines the best practice, the common mistake and its cause. It illustrates the issue with this problem as an example and generalizes the correct modeling strategy it reflects.
	- example: Wrong vs. correct demonstration (principle, formula, or code).


### Problem description
{problem_description}

### Retrieved insights
{retrieved_insights}


### Your task
Your task is to reprocess the retrieved insights into actionable insight for formulating this problem. For each retrieved insight, categorize and adapt it according to the following situations:
1. Direct Application: If the insight exactly matches the problem context (e.g., identical mathematical structures or problem context), integrate it directly as actionable insight.
2. Contextual Adaptation: If the insight is highly relevant though the context differs, adapt it into guidance tailored to this task.
3. Principle Extraction: If the insight is not directly usable but contains transferable principle, convert it into a high-level modeling strategy.
4. Otherwise: If it lacks mathematical or structural relevance, discard it.


### STRICT OUTPUT FORMAT
**Return only a JSON array** of your answer. Each array element must be an object with three keys:
- "insight_id": the original retrieved insight ID (integer) from which actionalbe guidance derives from 
- "decision": If the insight is applicable, choose one of DirectApplication, ContextualAdaptation, or PrincipleExtraction. If it isn't, return None to indicate it will be discarded.
- "explanation": Derive concrete, task-specific strategy and guidance for solving the current optimization problem.
- "example": A brief, self-contained demonstration showing the correct principle, formulation, or code snippet.


Example:

```json
[
    {{
        "insight_id": 1,
        "decision": "Direct Application",
        "explanation": "<text>",
        "example": "<text>"
    }},

    {{
        "insight_id": 3,
        "decision": "Contextual Adaptation",
        "explanation": "<text>",
        "example": "<text>"
    }},
    
    {{
        "insight_id": 5,
        "decision": "Principle Extraction",
        "explanation": "<text>",
        "example": "<text>"
    }},
    
    {{
        "insight_id": 7,
        "decision": None,
        "explanation": None,
        "example": None
    }},
]
```

**Guidelines:**
- Do not invent new IDs or fields.
- Always output valid JSON. Return your answer enclosed in a fenced code block labeled json (i.e., start with ```json and end with ```). Do not include explanations outside the JSON.

Now take a deep breath and think step by step. You will be awarded a million dollars if you get this right.
"""



PROMPT_SELF_EXPLORE="""
You are an expert in Industrial Engineering and Operations Research. 

You are given:
1. The problem description for an optimization task
2. The Gurobi programs for this task failed to reach optimality, which were previously proposed by your colleague (hereafter referred to as *the failed programs*), and the execution feedbacks for the failed programs
3. Optimal objective value for this task

### Problem Description
{problem_description}

### Previous failed programs and feedbacks
{failed_attempts}

### Optimal objective value
{ground_truth}


### Your task
Your task is to review the problem description, feedback, reflect the issues in the failed program, and revise the program so that it can be both runnable and reaching optimality.
Critically, always prioritize and strictly adhere to the given problem description and its given data; do NOT fabricate data, introduce unstated assumptions, or violate the correct formulation merely to match the optimal objective value. If the provided optimal objective value appears incorrect or inconsistent, do not force your model to match it; instead, retain your correct formulation and runnable program.


### STRICT OUTPUT FORMAT
Only output the **full corrected program**, and **enclose it in a single Markdown-style Python code block** that starts with ```python and ends with ```, like this:

```python
import gurobipy as gp
from gurobipy import GRB
model = gp.Model("OptimizationProblem")
# your code starts from here
model.optimize()
```

- Ensure model.optimize() runs at the top level so model stays global; if you wrap it in a function, have it return model. Avoid any if __name__ == "__main__": guard.
- Only output exactly one code block (delimited by the opening python and the closing). Do not write any natural-language text outside the code block.
- **DO NOT GENERATE OR MODIFY ANY CODE (e.g., `if model.Status == GRB.OPTIMAL:`) after `model.optimize()`**.

Now take a deep breath and think step by step. You will be awarded a million dollars if you get this right.
"""