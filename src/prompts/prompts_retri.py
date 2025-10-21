PROMPT_QUICK_MATCH_MODEL = """
You are an expert in Industrial Engineering and Operations Research. Your goal is to identify relevant insights that guide the construction of precise, error-free mathematical optimization models, inform sound domain-specific techniques, and prevent formulation pitfalls.

You are given:
1. A problem description.
2. Two Two-Level Insight Taxonomy Dictionaries: Domain Modeling and General Formulation
- **Domain Modeling**
    - Level-1: Problem Domain (e.g., "Network Flow")
    - Level-2: Domain-specific Technique/Principle (e.g., "Flow Conservation")
- **General Formulation**
    - Level-1: Formulation Component (e.g., "Variable Definition")
    - Level-2: Specific Aspect/Pitfall (e.g., "Continuous vs. Discrete Confusion")

### Problem description
{problem_description}

### Taxonomy Dictionary for Domain Modeling
{domain_taxo}

### Taxonomy Dictionary for General Formulation
{formu_taxo}


### YOUR TASK
1. Carefully read the problem description, then:
    - Identify which problem domain(s) and modeling techniques are involved.
    - Analyse potential formulation pitfalls the problem may entail.

2. Review the two taxonomy dictionaries carefully (both Level-1 and Level-2 labels with definitions and conditions) and return **only** the Level-1 and Level-2 labels that could apply to the current problem. You must ensure that every label you list exists in the provided taxonomy dictionary exactly as written.


### STRICT OUTPUT FORMAT
Return only a JSON object with the exact structure below:
- Outer keys = "Domain Modeling" or "General Formulation"
- Values = dictionaries whose keys are Level-1 labels from the taxonomy  
- Each Level-1 key's value = a list of one or more Level-2 labels from the taxonomy

**Note:** You may list multiple Level-1 and Level-2 labels if applicable. Example:

{{
    "Domain Modeling": {{
        "Resource Allocation": ["Multi-Commodity Flow", "Capacity/Resource Balance Equations"]
    }},
    "General Formulation": {{
        "Variable Definition": ["Continuous vs. Discrete Confusion"], 
        "Constraint Formulation": ["Incorrect Relational Operators"]
    }}
}}

If no taxonomy labels apply to the problem, it's perfectly fine. Return exactly:
{{}}

**Guidelines:** 
- The returned JSON must be valid—parsable by any standard JSON parser without modification.
- Ensure all returned labels exactly match those in the provided taxonomy dictionary.
- Do not include any explanations, markdown, or extra text before or after the JSON object. 

Take a deep breath and think step by step. You will be awarded a million dollars if you get this right.
"""


PROMPT_FULL_CHECK_MODEL = """
You are an expert in Industrial Engineering and Operations Research. 

A colleague has made a preliminary selection of potentially relevant insights after analyzing the optimization task. Your job is to **carefully evaluate** each candidate and decide whether it truly applies. 

You are given:
1. A problem description.
2. A collection of insights, each with: 
    - taxonomy: the classification of modeling, formulation or code implementation it lies in
    - condition: Statement of both when the insight **does apply (applicability condition)** and when it **does not (inapplicability condition)**, grounded in problem-specific context and broader modeling situations to prevent misuse.

### Problem description
{problem_description}

### Candidate insights
{candidate_insights}


### YOUR TASK
1. Carefully read the problem description, then:
    - Identify which problem domain(s) and modeling techniques are involved.
    - Analyse potential formulation pitfalls the problem may involve.
2. Evaluate each candidate insight one by one. **Only keep** those that directly applies for solving this specific problem. Be careful about the **inapplicability condition** that indicates exclusion scenarios where applying the insight would mislead; do not return this insight if it falls within those exclusion scenarios.
3. Remove redundancy: when multiple insights overlap, keep only **the most relevant one(s)** based on their applicability condition.
4. Use the **exact insight_id** provided with each candidate; do not invent new IDs.

### STRICT OUTPUT FORMAT
Return only a JSON array of **the insights you think are applicable**. Do not include explanations, markdown, or extra text.  
Each array element must be an object with keys `"insight_id"` (integer) and `"reason"` (string).  
Example:

```json
[
    {{"insight_id": 1, "reason": "<1-2 sentences>"}},
    {{"insight_id": 5, "reason": "<1-2 sentences>"}}
]
```

**If no insight is applicable, return exactly:**
[]

**Guidelines:** 
- The output must be valid JSON—parsable by any standard JSON parser without modification.
- Only keep those that **clearly apply** to this specific problem. **Do NOT include** any insight that is not applicable.
- Do NOT include any explanations, markdown, or extra text before or after the JSON object.

Now Take a deep breath and think step by step. You will be awarded a million dollars if you get this right.
"""

PROMPT_QUICK_MATCH_CODE = """
You are an expert in Industrial Engineering and Operations Research. Your goal is to identify relevant insights that can give guidance to produce the complete and runnable gurobipy code that strictly adheres to the given mathematical model of the optimization problem.

You are given:
1. A problem description.
2. A complete mathematical model with parameters, variables, constraints, objective function formulated by your colleague.
3. A two-level insight taxonomy dictionary: Level-1 label = Coding Area (e.g., "Solver & API Syntax"); Level-2 label = Specific Aspect/Issue (e.g., "Library Import/Reference Errors").

### Problem description
{problem_description}

### Mathematical model
{mathematical_model}

### Insight taxonomy dictionary
{taxo}


### YOUR TASK
1. Carefully read the problem description and the provided mathematical model.
2. Review the taxonomy dictionary carefully (both Level-1 and Level-2 labels with definitions and conditions) and return **only** the Level-1 and Level-2 labels that could apply to help write the code to solve the mathematical model of the current problem. You must ensure that every label you list exists in the provided taxonomy dictionary exactly as written.

### STRICT OUTPUT FORMAT
Return only a JSON object with the exact structure below:
- Outer keys = "Code Implementation"
- Values = dictionaries whose keys are Level-1 labels from the taxonomy  
- Each Level-1 key's value = a list of one or more Level-2 labels from the taxonomy  

**Note:** You may list multiple Level-1 and Level-2 labels if applicable. Example:

{{
    "Code Implementation": {{
        "Solver & API Syntax": ["Library Import/Reference Errors"]
    }}
}}

If no taxonomy labels apply to the current mathematical model, it's perfectly fine. Return exactly:
{{}}

**Guidelines:** 
- The returned JSON must be valid—parsable by any standard JSON parser without modification.
- Ensure all returned labels exactly match those in the provided taxonomy dictionary.
- Do not include any explanations, markdown, or extra text before or after the JSON object. 

Take a deep breath and think step by step. You will be awarded a million dollars if you get this right.
"""


PROMPT_FULL_CHECK_CODE = """
You are an expert in Industrial Engineering and Operations Research. 

A colleague has made a preliminary selection of potentially relevant insights. Your job is to **carefully evaluate** each candidate and decide whether it truly applies. 
You are given:
1. A problem description.
2. A complete mathematical model with parameters, variables, constraints, objective function generated by your colleague.
3. A collection of insights, each with: taxonomy—the classification of modeling, formulation or code implementation it lies in; and condition—the problem-specific context and broader modeling situations in which the insight should apply to avoid mistakes (i.e., its applicability condition).

### Problem description
{problem_description}

### Mathematical model
{mathematical_model}

### Candidate insights
{candidate_insights}

### YOUR TASK
1. Carefully read the problem description and the provided mathematical model.
2. Before writing the code to solve the mathematical model with Gurobi, evaluate each candidate insight one by one. Keep only those that are **directly applicable to avoiding code-implementation errors**.
3. Remove redundancy: when multiple insights overlap, keep only **the most relevant one or several** based on its applicability condition.
4. Use the **exact insight_id** provided with each candidate; do not invent new IDs.

### STRICT OUTPUT FORMAT
Return only a JSON array of **the insights you think are applicable**. Do not include explanations, markdown, or extra text.  
Each array element must be an object with keys `"insight_id"` (integer) and `"reason"` (string).  
Example:

```json
[
    {{"insight_id": 1, "reason": "<1-2 sentences>"}},
    {{"insight_id": 5, "reason": "<1-2 sentences>"}}
]
```

**If no insight is applicable, it's perfectly fine. Return exactly:**
[]

**Guidelines:** 
- The output must be valid JSON—parsable by any standard JSON parser without modification.
- Only keep those that **clearly apply** to this specific problem. **Do NOT include** any insight that is not applicable.
- Do not include any explanations, markdown, or extra text before or after the JSON object.

Now Take a deep breath and think step by step. You will be awarded a million dollars if you get this right.
"""