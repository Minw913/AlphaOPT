PROMPT_DIAGNOSE_ISSUES="""
You are an expert in Industrial Engineering and Operations Research.

You are given:
1. A problem description for the optimization task
2. A mathematical model proposed by your colleague which failed to yield an optimal solution when solved with the Gurobi optimizer (hereafter referred to as *the failed mathematical model*)
3. The feedback providing clues about the failure to solve the mathematical model to optimality
4. The gold-standard program, which embodies the correct mathematical formulation of the optimization task


### Problem description
{problem_description}


### The failed mathematical model
Note: the model is written in LaTeX and presented in a plain-text code block (```)
{failed_formulation}


### The feedback
{feedback}


### The gold-standard program
{correct_program}


### Your task

Step 1: Compare the failed mathematical model with the correct one embodied in the gold-standard program, and identify all formulation issues that prevent optimality. Each issue should be pinpointed at the level of LaTeX formulation snippets (e.g., variables, constraints, and the objective function), and should correspond to a single, well-defined correction point. Note that variable names in the proposed model may differ from those in the gold-standard program, so please align them carefully based on the problem specification.

Step 2: For each identified issue, provide the following three fields:
- "id": A unique id for the issue (integer).
- "issue": A concise description of the issue.
- "evidence": The evidence showing where the issue occurs, including the excerpt from the failed mathematical model (mark as #wrong) and the corresponding excerpt from the gold-standard program (mark as #correct).

Step 3: Minimize overlap by reporting **independent, root-cause issues**. If multiple defects share the same fix point or strategy, merge them into a single composite issue. If several issues are upstream/downstream symptoms of the same root cause (i.e., they would be fixed by the same correction), consolidate them into one composite issue.


### STRICT OUTPUT FORMAT
**Return only a JSON array** of your answer. Each array element must be an object with keys `"id"`, `"issue"` and `"evidence"`.

Example:

```json
[
    {{"id": 1,"issue": "...", "evidence": "..."}},
	{{"id": 2,"issue": "...", "evidence": "..."}}
]
```

**Guidelines:**
- Make sure to identify **distinct and independent issues** (e.g., missing constraints, wrong variable bounds, or incorrect objective formulation). 
- Do NOT include issues that do not directly affect the model's ability to reach optimality.
- Only output the JSON array. DO NOT include any explanations, markdown, or extra text before or after the JSON array.

Now take a deep breath and think step by step. You will be awarded a million dollars if you get this right.
"""




PROMPT_INS_POS_NEG="""
You are an expert in Industrial Engineering and Operations Research.

You are given:
1. A problem description for the optimization task
2. A mathematical model proposed by your colleague which failed to yield an optimal solution when solved with the Gurobi optimizer (hereafter referred to as *the failed mathematical model*)
3. A diagnostic report on the proposed mathematical model, identifying **all formulation issues** that prevent optimality.
4. A collection of insights your colleague previously consulted to generate the mathematical model, each insight includes:
    - insight_id: the unique ID for this insight
    - condition: the problem-specific context and broader modeling situations in which the insight should apply to avoid mistakes (i.e., its applicability condition)
    - explanation: the description of the pitfalls and guidance for the proper principle/practice
    - example: the demonstration showing wrong vs. correct version (principle, formula, or code snippet)


### Problem description
{problem_description}


### The failed mathematical model
Note: the model is written in LaTeX and presented in a plain-text code block (```), with brief comments indicating the corresponding insight_id and how it helps the specific formulation.
{failed_formulation}


### The diagnosed issues
{diagnosed_issues}


### A collection of insights
{retrieved_insights}


### Your task
Step 1: Carefully review the problem description, and analyse the failed mathematical model with the issues identified in the diagnostic report.

Step 2: Examine the collection of insights and the corresponding annotations in the proposed model that indicate how each insight was implemented. 

Step 3: For each insight, determine its correctness by comparing with the gold-standard program, then evaluate its implementation impact (whether it leads to any of the identified issues):
    1. Assign **"positive"** label when:
        The insight is correct and adopted by the model, and it contributed to a correct formulation (helped achieve the right result).

    2. Assign **"invalid"** label when:
        The insight is correct but not adopted; if adopted, it would have helped produce the correct formulation and resolve identified issues.

    3. Assign **"negative"** label when:
        - The insight is wrong and adopted, thereby directly causing an identified issue;
        - The insight is wrong and not adopted, yet it provides inapplicable/misleading guidance that poses a risk of errors.
        
    4. Assign **"irrelevant"** label when:
        The insight is irrelevant to the mathematical modeling in this optimization task and did not affect your colleague's formulation.

Suggested decision order: 
- Judge correct vs. wrong.
- Check adopted vs. not adopted.
- Assess impact (enabled correctness, caused issues, could resolve issues, risk/irrelevant).
- Map to one of the four labels.

Step 4: Record the insight_id and the assigned label. Cite concrete evidence from the problem description and diagnosed issues, and justify the labeling. **Clearly explain the mapping between each insight and the formulation issues.**

### STRICT OUTPUT FORMAT
**Return only a JSON array** of your answer in Step 4. Each array element must be an object with keys `"insight_id"` (integer), `"state"`("postive" or "negative") and `"evidence"` (string).

Example:

```json 
[       
	{{"insight_id": 1, "state": "positive", "evidence": "<text>"}},     
	{{"insight_id": 5, "state": "negative", "evidence": "<text"}} 
] 
```

**Guidelines:**
- Make sure to identify and output each insight_id with its state. Do NOT miss any insight id in the given collection.
- Only output the JSON array. DO NOT include any explanations, markdown, or extra text before or after the JSON array.

Now take a deep breath and think step by step. You will be awarded a million dollars if you get this right.
"""



PROMPT_RETRI_LABEL="""
You are an expert in Industrial Engineering and Operations Research. 

You are given:
1. A problem description for the optimization task
2. A mathematical model proposed by your colleague that failed to yield an optimal solution when solved with the Gurobi optimizer (hereafter referred to as *the failed mathematical model*)
3. A diagnostic report on the proposed mathematical model, identifying the specific issue that prevents optimality


### Problem description
{problem_description}


### The failed mathematical model
(Note: the model is written in LaTeX and presented in a plain-text code block (```))
{failed_formulation}


### The diagnosed issue
{one_diagnosed_issue}


### Your task
Step 1: Carefully review the problem description, and analyse the failed mathematical model with the issue identified in the diagnostic report.

Step 2: Given the full taxonomy dictionaries of existing insights stored in library (shown below), determine the potential level-1 and level-2 label(s) under which relevant useful insights may be found for **resolving the identified issue**.

Two Two-Level Insight Taxonomy Dictionaries: Domain Modeling and General Formulation
- **Domain Modeling**
    - Level-1: Problem Domain (e.g., "Network Flow")
    - Level-2: Domain-specific Technique/Principle (e.g., "Flow Conservation")
- **General Formulation**
    - Level-1: Formulation Component (e.g., "Variable Definition")
    - Level-2: Specific Aspect/Pitfall (e.g., "Continuous vs. Discrete Confusion")
    

### Taxonomy Dictionary for Domain Modeling
{domain_taxo}

### Taxonomy Dictionary for General Formulation
{formu_taxo}


### STRICT OUTPUT FORMAT
**Return only a JSON object** of your analysis result in Step 2, with the exact structure below:
- Outer keys = "Domain Modeling" or "General Formulation"
- Values = dictionaries whose keys are Level-1 labels from the taxonomy  
- Each Level-1 key's value = a list of one or more Level-2 labels from the taxonomy

Note:
- You may assign multiple level-1 and level-2 labels to the issue only when you think they are all potentially applicable.
- If no applicable labels exist the issue, simply set its "matched_label(s)" value to null.

Example 1 - Multiple applicable level-1 and level-2 labels:
{{
    "Domain Modeling": {{
        "Production Planning": ["Inventory Balance Equations", "Time-Indexed Variables"],
        "Resource Allocation": ["Capacity/Resource Balance Equations"]
    }}
{{

Example 2 - Multiple applicable level-1 and level-2 labels from both Domain Modeling and General Formulation:
{{
    "Domain Modeling": {{
        "Facility Location": ["Fixed Charge (Big-M Linking)"]
    }},
    "General Formulation": {{
        "Constraint Formulation": ["Big-M & Indicator Variables"]
    }}
{{

Example 3 - No taxonomy labels apply to the issue:
{{}}


**Guidelines:**
- You must ensure that every label you list exists in the provided taxonomy dictionary exactly as written.
- Only output the JSON object. DO NOT include any explanations, markdown, or extra text before or after the JSON array.

Now take a deep breath and think step by step. You will be awarded a million dollars if you get this right.
"""




PROMPT_RETRI_INS="""
You are an expert in Industrial Engineering and Operations Research. 

You are given:
1. A problem description for the optimization task
2. A mathematical model proposed by your colleague which failed to yield an optimal solution when solved with the Gurobi optimizer (hereafter referred to as *the failed mathematical model*)
3. A diagnostic report on the proposed mathematical model, identifying the specific issue that prevents optimality
4. A collection of insights. Each insight includes:
    - insight_id: the unique identifier of the insight
    - taxonomy: the classification of the modeling/formulation/code-implementation aspect it pertains to
    - condition: the problem-specific context and broader modeling situations in which the insight applies (its applicability condition)


### Problem description
{problem_description}


### The failed mathematical model
Note: the model is written in LaTeX and presented in a plain-text code block (```)
{failed_formulation}


### The diagnosed issue
{one_diagnosed_issue}


### Candidate insights
{candidate_insights}


### Your task
Step 1: Carefully review the problem description, and analyse the failed mathematical model with the issue identified in the diagnostic report.

Step 2: Evaluate each candidate insight individually. Retain only those that directly apply to **resolving the identified issue** in the diagnostic report. For every insight, cite concrete evidence from the problem description and diagnosed issues, and justify how the insight helps fix the specific modeling issue.

Step 3: Rank the applicability of the selected insights in descending order.


### STRICT OUTPUT FORMAT
**Return only a JSON array** of your result from Step 3. Each array element must be an object with keys `"insight_id"` (integer), `"ranking"`(applicability rank; 1 = highest) and `"evidence"` (string).  
Example:

```json
[
    {{"insight_id": 1, "ranking": 1, "evidence": "<text>"}},
    {{"insight_id": 5, "ranking": 2, "evidence": "<text>"}},
	{{"insight_id": 7, "ranking": 3, "evidence": "<text>"}}
]
```

**Guidelines:**
- Output only the insights that apply to the identified issue(s).
- Only output the JSON array. DO NOT include any explanations, markdown, or extra text before or after the JSON array.

Now take a deep breath and think step by step. You will be awarded a million dollars if you get this right.
"""




PROMPT_VALIDATE_ISSUES="""
You are an expert in Industrial Engineering and Operations Research.

You are given:
1. A problem description for the optimization task
2. A mathematical model proposed by your colleague which failed to yield an optimal solution when solved with the Gurobi optimizer (hereafter referred to as *the failed mathematical model*)
3. A diagnostic report on the proposed mathematical model, identifying **all formulation issues** that prevent optimality.
4. The regenerated mathematical model, which was recreated by your colleague based on the diagnostic report.


### Problem description
{problem_description}


### The failed mathematical model
Note: the model is written in LaTeX and presented in a plain-text code block (```)
{failed_formulation}


### The diagnosed issues
{diagnosed_issues}


### The regenerated mathematical model
{new_formulation}


### Your task
Step 1: Carefully review the problem description, and analyse the failed mathematical model with the issues identified in the diagnostic report.

Step 2: Carefully review the regenerated mathematical model and compare it with the failed model to determine whether each previously identified issue has been resolved. For each issue, report:
- "id": A unique id for the issue (integer).
- "status": "solved" or "unsolved". 
- "evidence": The evidence supporting the status, including the relevant excerpt from the failed model and the corresponding excerpt from the regenerated model.


### STRICT OUTPUT FORMAT
**Return only a JSON array** of your answer. Each array element must be an object with keys `"id"`, `"status"` and `"evidence"`.

Example:

```json
[
    {{"id": 1, "status": "solved", "evidence": "<text>"}},
	{{"id": 2, "status": "unsolved", "evidence": "<text>"}}
]
```

**Guidelines:**
- Make sure to identify and output each distinct issue. Do NOT miss any issue id in the given diagnostic report.
- Only output the JSON array. DO NOT include any explanations, markdown, or extra text before or after the JSON array.

Now take a deep breath and think step by step. You will be awarded a million dollars if you get this right.
"""




PROMPT_PROGRAM_DIAG="""
You are an expert in Industrial Engineering and Operations Research. 

You are given:
1. A Gurobi program failed to execution (hereafter referred to as *the failed program*)
2. The execution error message for the failed program


### The failed program
{failed_program}


### Error message
{feedback}


### Your task
Your task is to review the execution error message, identify the issues in the failed program that caused the error, and revise the program so that it can run successfully.

For **each issue**: 
- Explain the issue in a short comment
- Comment out the wrong code line(s) using `# wrong attempt: ...`
- Write the corrected code right after the comment
- Wrap each issue block with exactly one `#===` line above and below.

**Example format:**
#===
# <issue explanation>
# <commented-out incorrect code>
<corrected code>
#===

**Leave all other code unchanged.**


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
- **DO NOT MODIFY ANY CODE after the line model.optimize()**.

Now take a deep breath and think step by step. You will be awarded a million dollars if you get this right.
"""