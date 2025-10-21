PROMPT_INS_FROM_FORMU = """
You are an expert in Industrial Engineering and Operations Research teaching graduate students to avoid modeling-and-coding mistakes in solving optimization problems.

You are given:
1. A problem description for the optimization task
2. A mathematical model proposed by your colleague which failed to yield an optimal solution when solved with the Gurobi optimizer (hereafter referred to as *the failed mathematical model*)
3. The gold-standard program, which embodies the correct mathematical formulation of the optimization task


### Problem description
{problem_description}


### The failed mathematical model
(Note: the model is written in LaTeX and presented in a plain-text code block (```))
{failed_formulation}


### The gold-standard program
{correct_program}


### Your task
Step 1: Compare the failed mathematical model with the correct mathematical model embodied in the gold-standard program to identify issues that prevent optimality. Note that variable names in the proposed model may differ from those in the gold-standard program. Please align them carefully based on the problem specification.

Step 2: Using the insight taxonomy dictionaries provided below, extract one or more new insights, which should be a distinct and concise lesson derived from a specific issue identified in the failed mathematical model relative to the gold-standard program.

Each new insight must contain exactly four fields:

1) **taxonomy** — choose **exactly one** of the two aspects:
    - **Domain Modeling**: Level-1 = Problem Domain (e.g., "Network Flow"); Level-2 = Specific Technique (e.g., "Flow Conservation").
    - **General Formulation**: Level-1 = Formulation Component (e.g., "Variable Definition"); Level-2 = Specific Aspect/Pitfall (e.g., "Continuous vs. Discrete Confusion").

    Taxonomy rule (nested-dict): `{{ Level-1 : {{ Level-2 : null | {{ "definition": "...", "condition": "..." }} }} }}`
    - The taxonomy MUST always be expressed as a **three-level nested dict**: Level-1 → Level-2 → (null or a dict with "definition"/"condition").
    - Pick **exactly one** Level-1 (existing key).
    - Pick **one or more** Level-2 under that Level-1 (existing key or keys).
    - For an existing Level-2, set its value to null.
    - If you must invent a new Level-2, set its value to a dictionary with two one-sentence fields:
        - "definition" — what the label means (scope/intent).
        - "condition" — when to apply the label (a general trigger grounded in the problem description or in the defining features of the problem domain).
    - If you must invent a new Level-1, include ≥1 Level-2 under it; each invented Level-2 must provide both "definition" and "condition".

2) **condition** — Write it as a trigger explicitly grounded in the problem description or in the defining features of the problem domain. First state the general situation, then use this problem as an example. **Use the pattern below**, and keep it strictly non-prescriptive: do not give any solution, advice or decision:
"This insight applies when ... For example, when the problem statement mentioned ...". 

3) **explanation** — A brief and self-contained description that specifies, under this condition, what the best practice is, what the common mistake is and its cause. First, use this problem as an example to illustrate; Then, appropriately generalize the correct modeling strategy it reflects, if applicable.
**Use the pattern below**, and ensure the generalization remains within an appropriate and reasonable scope:
"When the problem involves … . The best practice is … . A common mistake is … , which happens because … . More generally, this reflects that … ." 

4) **example** — A brief, self-contained demonstration showing wrong vs. correct version (principle, formulation, or code snippet). Clearly mark them as '# Wrong' and '# Correct'.

### Taxonomy Dictionaries
**Domain Modeling**
{domain_taxo}

**General Formulation**
{formulation_taxo}

### STRICT OUTPUT FORMAT
Return a single JSON **array** of insight objects. No text before/after. Example with two insights (but not must be two):

[
    {{
        "taxonomy": {{
            "Domain Modeling": {{
                "Network Flow": {{
                    "<New Label If Necessary>": {{ "definition": "<one sentence>", "condition": "<one sentence>" }}
                }}
            }}
        }},
        "condition": "<text>",
        "explanation": "<text>",
        "example": "<text>"
    }},

    {{
        "taxonomy": {{
            "General Formulation": {{
                "Variable Definition": {{
                    "Continuous vs. Discrete Confusion": null
                }}
            }}
        }},
        "condition": "<text>",
        "explanation": "<text>",
        "example": "<text>"
    }}
]

**Guidelines**:
- Output as many **distinct, non-overlapping** insights as needed.
- Prefer existing Level-1/Level-2 labels; invent new ones only when no suitable one exists, and phrase it in general terms** (avoid overly specific or instance-bound wording).
- **Be precise in stage selection**—use **Domain Modeling** for domain-specific techniques that arise only within specific problem families (e.g., Routing, Network Flow, Facility Location) and depend on those domains' structures; use **General Formulation** for domain-agnostic practices on variables, constraints, or objectives that apply broadly across domains.

Now take a deep breath and think step by step. You will be awarded a million dollars if you get this right.
"""

PROMPT_INS_FROM_PROGRAM = """
You are an expert in Industrial Engineering and Operations Research teaching graduate students to avoid coding mistakes.

You are given:

### Corrected program with #=== fix blocks delimited by:
    #===
    # <brief issue comment>
    # <commented-out erroneous code>
    <corrected code>
    #===
{corrected_program}


### Your Task  
For each `#===` fix block in the corrected program, extract one or more **new insights**—each a distinct, concise lesson distilled from the commented-out coding mistake.

Each new insight must contain exactly four fields:

1) **taxonomy** — the classification of specific aspect/issue it applies to.

    Taxonomy rule (nested-dict): `{{ Level-1 : {{ Level-2 : null | {{ "definition": "...", "condition": "..." }} }} }}`
    Level-1 = Coding Area (e.g., "Solver & API Syntax"); Level-2 = Specific Aspect/Issue (e.g., "Library Import/Reference Errors").

    - Ensure that the outermost dictionary key is always "Code Implementation"
    - Pick **exactly one** Level-1 (existing key).
    - Pick **one or more** Level-2 under that Level-1 (existing key or keys).
    - For an existing Level-2, set its value to null.
    - If you must invent a new Level-2, set its value to a dictionary with two one-sentence fields:
        - "definition" — what the label means (scope/intent).
        - "condition" — when to apply the label (a general trigger explicitly grounded in the mathematical model).
    - If you must invent a new Level-1, include ≥1 Level-2 under it; each invented Level-2 must provide both "definition" and "condition".

2) **condition** — Write it as a trigger explicitly grounded in the mathematical model. State the general modeling pattern, then use this model as an example. Use the pattern: "This insight applies when the mathematical model contains... For example, when the formulation included...". Keep it strictly non-prescriptive: do not give any solution, advice or decision.

3) **explanation** — A brief and self-contained description that specifies, under this condition, what the best practice is, what the common mistake is and its cause. First, use this problem as an example to illustrate; Then, appropriately generalize the correct modeling strategy it reflects, if applicable.
**Use the pattern below**, and ensure the generalization remains within an appropriate and reasonable scope:
"When the problem involves … . The best practice is … . A common mistake is … , which happens because … . More generally, this reflects that … ." 

4) **example** — A brief, self-contained demonstration showing wrong vs. correct version (principle, formulation, or code snippet). Clearly mark them as '# Wrong' and '# Correct'.

### Taxonomy Dictionary
{code_taxo}


### STRICT OUTPUT FORMAT
Return a single JSON **array** of insight objects. No text before/after. Example with two insights (but not must be two):

[
    {{
        "taxonomy": {{
            "Code Implementation": {{
                "Solver & API Syntax": {{
                    "Library Import/Reference Errors": null
                }}
            }}
        }},
        "condition": "<text>",
        "explanation": "<text>",
        "example": "<text>"
    }},

    {{
        "taxonomy": {{
            "Code Implementation": {{
                "Data I/O & Validation": {{
                    "KeyError & Index Mismatch": null
                }}
            }}
        }},
        "condition": "<text>",
        "explanation": "<text>",
        "example": "<text>"
    }},
]

**Guidelines**:
- Output as many **distinct, non-overlapping** insights as needed.
- Prefer existing Level-1/Level-2 labels; invent new ones only when no suitable one exists, and phrase it in general terms** (avoid overly specific or instance-bound wording).

Now take a deep breath and think step by step. You will be awarded a million dollars if you get this right.
"""


PROMPT_CONDUCT_MERGE="""
You are an expert in Industrial Engineering and Operations Research. 

You are given a collection of insights, which are used to guide better solve the optimization tasks. Each insight with five fields:
	- id: A unique id for the insight.
	- taxonomy: The categorical labels of this insight, indicating its orientation and what specific aspect it addresses.
    - condition: Trigger specifying when the insight applies, grounded in problem description/domain features. States the general situation, then illustrates with the specific problem.
    - explanation: Under this condition, the description outlines the best practice, the common mistake and its cause. It illustrates the issue with this problem as an example and generalizes the correct modeling strategy it reflects.
    - example: Wrong vs. correct demonstration (principle, formula, or code).


## A collection of insights
{candidate_insights}

## Your task 
Step 1: Review all insights in the collection. Compare their taxonomy, condition, explanation, and examples, and infer the underlying modeling principles and the pitfalls they target.

Step 2: Identify insights with logically equivalent principles/methods in Operations Research, even if phrased differently or arising in different scenarios. When appropriate, merge them into a single canonical insight by consolidating applicability conditions, unifying the explanation, and keeping representative examples. Do **not** merge insights that address materially different mistakes.

Step 3: If you decide to merge, select a taxonomy. If the merged insights have exactly the same taxonomy, keep the original taxonomy. If they differ, choose the taxonomy that best fits the merged insight. **Please ensure that the taxonomy content and format remain exactly the same as the original.**


## STRICT OUTPUT FORMAT
**Only Return a single JSON array of** with ONLY one of the following structures:

1) If you decide NOT to merge then output:

```json
[]
```

2) Merge any subset(s) of them into one then output:

```json
[
    {{
        "merged_ids": [/* ids you are about to merged */],
        "reason": "<why merging reduces redundancy>",
        "taxonomy": {{
            "General Formulation": {{
                "Variable Definition": {{
                    "Continuous vs. Discrete Confusion": null
                }}
            }}
        }},
        "condition": "<State the shared trigger context grounded in the problem description or domain features. Use the pattern: 'This insight applies when … For example, when the problem statement mentioned …'. Keep it strictly non-prescriptive—no solution/advice/decision.>",
        "explanation": "<Unify the best modeling principle and typical pitfalls covered by this group. Remove overlap, keep only essentials. Use the pattern: 'When the problem involves, … . The best practice is … . A common mistake is … , which happens because … . More generally, this reflects that … .' >",
        "example": "<Integrate representative examples from the originals>"
    }}
    /* , {{ ... other merged group(s) with same structure if any... }} */
]
```

**Guidelines:**
- Always output valid JSON. Return your answer enclosed in a fenced code block labeled json (i.e., start with ```json and end with ```). Do not include explanations outside the JSON.
- Do not invent new IDs or fields.
- it is equally acceptable to keep all insights separate if differences matter. Only merge when it truly reduces duplication.
- If a single insight doesn't need to be merged with others, skip its insight_id. Do NOT output a "merged_ids" list that contains only that one insight_id.

Now Take a deep breath and think step by step. You will be awarded a million dollars if you get this right.
"""



PROMPT_ONLINE_MERGE="""
You are an expert in Industrial Engineering and Operations Research. 

You are given a new insight, a collection of existing insights in the library, which are used to guide better solve the optimization tasks. Each insight has four fields and each existing library insight has A unique insight_id.
	- taxonomy: The categorical labels of this insight, indicating its orientation and what specific aspect it addresses.
    - condition: Trigger specifying when the insight applies, grounded in problem description/domain features. States the general situation, then illustrates with the specific problem.
    - explanation: Under this condition, the description outlines the best practice, the common mistake and its cause. It illustrates the issue with this problem as an example and generalizes the correct modeling strategy it reflects.
    - example: Wrong vs. correct demonstration (principle, formula, or code).


## A new insight
{new_insight}


## Existing insights in the library
{existing_insights}


## Your task 
Step 1: Review the new insight and the existing insights in the library. Compare their condition, explanation, and examples, and infer the underlying modeling principles and the pitfalls they target.

Step 2: Determine whether the new insight is logically equivalent to one or more existing library insights—that is, it reflects the same underlying OR principle or method—even if expressed differently or observed in different scenarios. If so, merge them into a single canonical insight by consolidating applicability conditions, unifying the explanation, and retaining representative examples. Do not merge insights that address materially different mistakes.


## STRICT OUTPUT FORMAT
**Only Return a single JSON object of** with ONLY one of the following structures:

1) **If you decide NOT to merge then output an empty JSON object:**

{{}}


2) Merge the new insight with one or some or all of the existing library insights into one then output:

{{
    "merged_ids": [/* the insight_id values of the existing library insights you are about to merge with the new insight */],
    "reason": "<why merging reduces redundancy>",
    "condition": "<State the shared trigger context grounded in the problem description or domain features. Use the pattern: 'This insight applies when … For example, when the problem statement mentioned …'. Keep it strictly non-prescriptive—no solution/advice/decision.>",
    "explanation": "<Unify the best modeling principle and typical pitfalls covered by this group. Remove overlap, keep only essentials. Use the pattern: 'When the problem involves, … . The best practice is … . A common mistake is … , which happens because … . More generally, this reflects that … .' >",
    "example": "<Integrate representative examples from the originals>"
}}

**Guidelines:**
- **Always output valid JSON.** Return your answer enclosed in a fenced code block labeled json (i.e., start with ```json and end with ```). Do not include explanations outside the JSON.
- Do not invent new IDs or fields.
- it is equally acceptable to keep all insights separate if differences matter. Only merge when it truly reduces duplication.

Now Take a deep breath and think step by step. You will be awarded a million dollars if you get this right.
"""