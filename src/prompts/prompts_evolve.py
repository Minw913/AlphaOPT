PROMPT_INS_NEG="""
You are an expert in Industrial Engineering and Operations Research. 

To better solve optimization tasks, we maintain an experience library that provides insights on mathematical modeling and writing solver programs. Your colleague previously diagnosed an optimization task that failed to reach optimality (due to formulation or programming errors) and found the cause was that someone incorrectly followed an insight from the library. This happened because the insight's applicability condition was poorly written (too broad or lacking counterexamples), causing the task where it does not apply to be misclassified as applicable.

You are given:

1. The insight to be refined, consisting of three fields:
	- condition: A trigger specifying when the insight applies, grounded in problem description/domain features. It first states the general situation, then illustrates with the specific problem.
	- explanation: Under this condition, the description outlines the best practice, the common mistake and its cause. It illustrates the issue with this problem as an example and generalizes the correct modeling strategy it reflects.
	- example: Wrong vs. correct demonstration (principle, formula, or code).

2. The problem description of an optimization problem for which this insight your colleague thought is not applicable.

3. The reason given by your colleague why this insight is not applicable to that optimization problem.


## The original insight to be refined
{target_insight}

## Problem description
{desc}

## The reason that this insight is not applicable (optional)
{diag_evidence}


## Your tasks
Your tasks are:
- Re-verify whether the modeling or programming principle/strategy captured by this insight is indeed not applicable to this problem.
- If not applicable, write **inapplicability condition** for the insight so that others recognize it does not apply to this optimization problem and avoid incorrectly following its guidance.

Please follow these reasoning steps before give your answer:
Step 1: Carefully read the insight's applicability condition, explanation, and example to understand the problem context it targets and the modeling strategy it proposes.
Step 2: Carefully read the problem description that was incorrectly matched to this insight. Analyze whether your colleague's argument for inapplicability is sound, and confirm whether the insight's principle/strategy truly does not apply here.
Step 3: Based on Steps 1-2,
	- If the insight is indeed not applicable to this problem, write the insight's inapplicability conditions, strictly using the pattern:
"This insight does NOT apply when [general situation that negates the insight]. For example, when the problem statement mentions [concrete trigger grounded in the problem description or defining features indicating properties that conflict with the insight]."
	- If you conclude the insight actually applies and does not conflict with the correct formulation, do not make any modifications.


## STRICT OUTPUT FORMAT

**Only return a JSON object** under one of the following structures:

**If this insight is truly not applicable**, return:
{{
	"condition": "<write the inapplicability condition following the given instructions above>", 
	"reason": "<1-2 sentences explaining why you confirm this insight is not applicable to the given problem>"
}},

**If this insight is applicable, return exactly:**
{{}}

## Guidelines
- Always output valid JSON.  
- Do not include any explanation text outside the JSON.  

Now Take a deep breath and think step by step. You will be awarded a million dollars if you get this right.
"""


PROMPT_INS_UNR="""
You are an expert in Industrial Engineering and Operations Research. 

To better solve optimization tasks, we maintain an experience library that provides insights on mathematical modeling and writing solver programs. Your colleague previously diagnosed an optimization task that failed to reach optimality (due to formulation or programming errors) and found the cause was that someone failed to retrieve a useful insight from the library by its condition. This happened because the insight's applicability condition was poorly written—either too narrow in scope or lacking precise contextual triggers—causing the task to miss a suitable insight that could have led to the correct solution.

To better solve optimization tasks, we maintain an experience library that provides insights on mathematical modeling and writing solver programs. Your colleague diagnosed an optimization task that failed to reach optimality (due to formulation or programming errors) and found the cause was that a useful insight was not retrieved via its condition. The insight's applicability condition was poorly written—either too narrow in scope or lacking precise contextual triggers—causing the task to miss a suitable insight that could have guided the correct solution.

You are given:

1. The insight to be refined, consisting of three fields:
	- condition: A trigger specifying when the insight applies, grounded in problem description/domain features. It first states the general situation, then illustrates with the specific problem.
	- explanation: Under this condition, the description outlines the best practice, the common mistake and its cause. It illustrates the issue with this problem as an example and generalizes the correct modeling strategy it reflects.
	- example: Wrong vs. correct demonstration (principle, formula, or code).

2. The problem description of an optimization problem for which this insight your colleague thought is not applicable.

3. The reason given by your colleague why this insight is not applicable to that optimization problem.


## The original insight to be refined
{target_insight}

## Problem description  
{desc}

## The reason that this insight is applicable and useful  (optional)
{diag_evidence}


## Your tasks
Your tasks are:
- Re-verify whether the modeling or programming principle/strategy captured by this insight is indeed useful to this problem.
- If it is, write **a new applicability condition** for the insight that, compared with the original, is more specifically tailored to this problem's scenario, so that others will not overlook or fail to retrieve this insight when checking its applicability next time.

Please follow these reasoning steps before give your answer:
Step 1: Carefully read the insight's original applicability condition, explanation, and example to understand the problem context it targets and the modeling strategy it proposes.
Step 2: Carefully read the problem description that was failed to matched to this insight. Analyze whether your colleague's argument for applicability of this insight is sound, and confirm whether the insight's principle/strategy truly does apply here.
Step 3: Based on Steps 1-2,
	- If the insight is indeed applicable and useful to this problem, write the insight's new applicability condition using the pattern:
	"This insight applies when [general situation that warrants the insight]. For example, when the problem statement mentions [concrete trigger grounded in the problem description or defining features indicating properties that align with this insight]."
	- If you conclude the insight actually applies and does not conflict with the correct formulation, do not make any modifications.


## STRICT OUTPUT FORMAT

**Only return a JSON object** under one of the following structures:

**If this insight is truly applicable**, return:
{{
	"condition": "<write the new condition following the given instructions above>", 
	"reason": "<1-2 sentences explaining why you confirm this insight is not applicable to the given problem>"
}},

**If this insight is indeed not applicable, return exactly:**
{{}}

## Guidelines
- Always output valid JSON.  
- Do not include any explanation text outside the JSON.  

Now Take a deep breath and think step by step. You will be awarded a million dollars if you get this right.
"""


PROMPT_INS_REFINEMENT="""
You are an expert in Industrial Engineering and Operations Research. Your task is to **design multiple refinement strategies** for the **condition** of a given insight to improve its applicability in optimization tasks.

You are given:	
1. Original insight with three fields:
    - condition: Trigger specifying when the insight applies, grounded in problem description/domain features. It first States the general situation, then illustrates with the specific problem.
    - explanation: Under this condition, the description outlines the best practice, the common mistake and its cause. It illustrates the issue with this problem as an example and generalizes the correct modeling strategy it reflects.
    - example: Wrong vs. correct demonstration (principle, formula, or code).
	
2. Task-derived insight conditions:
- **Inapplicability conditions** of insights for **negative tasks** where prior use of the insight misled the modeling.
- **Applicability conditions** of insights for **unretrieved tasks** which should have retrieved these insights but were missed.

Applicability conditions for unretrieved tasks where the insight was relevant but not retrieved.

## The original insight to be refined
{original_insight}

### inapplicability conditions from negative tasks
{neg_conditions}

### applicability conditions from unretrieved tasks
{unr_conditions}


## Your tasks
Your goal is to **refine only the condition field of the original insight** so that it excludes as many **negative** tasks as possible, captures as many previously **unretrieved** tasks as possible, and still applies to the tasks covered by the original condition. To achieve this, consider the following four steps:

Step 1: **Consolidate inapplicability:** Merge all inapplicability conditions from negative tasks into a single, unified inapplicability condition. 
**Use the pattern:** 
	"This insight does NOT apply when [general situation that negates the insight]. For example, when the problem statement mentions [concrete trigger(s) grounded in the problem description or defining features indicating properties that conflict with the insight]."

Step 2: **Consolidate applicability**: Merge all applicability conditions from unretrieved tasks into a single, unified applicability condition.
**Use the pattern:** 
	"This insight applies when [general situation that warrants the insight]. For example, when the problem statement mentions [concrete trigger(s) grounded in the problem description or defining features indicating properties that align with this insight]."
	
Step 3: **Integrate into the insight's condition field**:
	- Preserve the original condition text verbatim.
	- Structure the field as three paragraphs using the pattern below.
	- Use the unified inapplicability condition as the second paragraph.
**Output using the pattern (three brief paragraphs)**:
	- First paragraph: original applicability condition
	- Second paragraph: unified applicability condition for unretrieved tasks: "This insight applies when …"
	- Third paragraph: unified inapplicability condition for negative tasks: "This insight does NOT apply when … "

Step 4: **Generate {path_k} distinct refinement strategies (paths):** Generate distinct refinement strategies (paths) for how you consolidate the insight conditions. For each path, write applicability/inapplicability using the required pattern in Step 3.
Examples: 
	- one broad rule: Merge multiple situations into a single general trigger that clearly and broadly covers the main applicable scenario.
	- short OR list: Provide a brief list of alternative triggers; if any one appears, the insight applies.
	- must-have pair: Require two key cues to occur together before the insight applies, reducing false positives.
	- info stated vs. missing: Apply only when the problem explicitly states the required details; treat omissions as not applicable.
	- helpful keywords: Anchor applicability on a small set of representative keywords or phrases that reliably signal the scenario.
	- narrow the scope: Add concise qualifiers to limit coverage to a well-bounded context so the insight applies only under those limits.

## STRICT OUTPUT FORMAT

**Only Return a JSON array** with the following structure:

[
    {{
        "path_id": 1,
        "strategy": "<1-2 sentences description of the refinement strategy>",
        "new_condition": "<rewritten condition string>"
    }},
    {{
        "path_id": 2,
        "strategy": "...",
        "new_condition": "..."
    }},
    ...
]

## Guidelines
- Always output valid JSON.  
- Do not include explanations outside the JSON.  
- Each path must be **semantically distinct**.  

Now Take a deep breath and think step by step. You will be awarded a million dollars if you get this right.
"""