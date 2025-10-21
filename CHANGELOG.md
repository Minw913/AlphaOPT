
------------
# 2025/08/27
------------
## Completed
### [Update] evolution.py, library_learning.py, iterative_library_learning_and_evolution.py,  llm_programmer.py, llm_retriever.py, llm_extractor.py

Update the workflow (formulation generation -> program generation) for programmer, retriever and extractor agents


------------
# 2025/08/14
------------
## Completed

### [Update_File_Name] src/: llm_programmer.py (llm_opt_generator.py), llm_diagnostic.py (llm_opt_diagnoser.py), llm_retriever.py (llm_lib_retrieval.py), llm_evolver.py (llm_lib_evolution.py)


### [Update] src/: llm_diagnostic.py, llm_extractor.py.(new), llm_retriever.py, experience_library.py

1. Decouple insight-extraction from `llm_diagnostic.py` and move it to `llm_extractor.py`; Implement insight extraction with taxonomy.

2. Change retrieval method from retrieving by condition to retrieving by taxonomy in `llm_retriever.py`.

3. Complete taxonomy dictionary load and update; retrieve by taxonomy features in `experience_library.py`.


### [Update] library_learning.py, iterative_library_learning_and_evolution.py

1. Integrate traininng and validation tasks with early stop.

2. Implement features for learning from uncertain success.



------------
# 2025/08/01
------------
## Completed

### [Add] scripts/: automate_case_study.py, tag_tasks.py, label_library.py

Automate the analysis of success and failure case studies, tag problem domain of tasks, label error type of library Insights.


### [Update] src/: llm_opt_generator.py, llm_opt_diagnoser.py, llm_lib_retrieval.py, llm_lib_evolution.py

1. Refine the llm output formats and parsing logic.

2. Refine the data and log tracking.



------------
# 2025/07/24
------------
## Completed

### [Update] src/: llm_opt_generator.py, llm_opt_diagnoser.py, llm_lib_retrieval.py

1. Refine the llm output formats and parsing logic.

2. Raised the optimality tolerance to 5e-3 (allowing for two-decimal rounding errors).

3. Add an execution-timeout mechanism to prevent long-running code for large-scale tasks from disrupting the training process.


### [Update] scripts/preprocess_dataset.ipynb

Integrate NLP4LP, IndustryOR, Logior datasets



------------
# 2025/07/17
------------
## Completed

### [Update] library_learning.py

Incorporate self-verify mechanism of new generated insights in each iteration. Discard new insights that cannot be verified on its source task.


### [Update] src/llm_lib_evolution.py

Implement MiniLM + HDBSCAN on library insights clustering, replacing the earlier method where insight merge groups were proposed by the LLM.


### [Update] src/llm_lib_retrieval.py

Implement insights applicability check after quickly matching by condition and reason on problem modeling.


### [Add] scripts/check_self_consistent_tasks.py, scripts/data_check_statistics.ipynb

Check data consistency and do correction by gemini-2.5-pro; Data check analysis.




------------
# 2025/07/10
------------
## Completed

### [Add] evaluation.py, eval_params.yaml

Implement the pipeline of evaluating library on test data.


### [Add] src/llm_lib_evolution.py

Implement pseudocode of library evolution (merge operations) by llm agents.


### [Add] library_evolution.py, library_learning.py

Implement multithread of iterative learning and evaluation pipeline.

`run_library_evolution()` in library_evolution.py: propose variants -> detect affected subtasks -> evaluate task success rate -> select the best variant.


### [Update] iterative_learning_and_evaluation.py

Implement pseudocode of the whole pipeline (retrieving -> learning -> evolving -> evaluating -> iterative doing).


### [Update] src/: utils.py, llm_opt_generator.py, llm_opt_diagnoser.py, llm_lib_retrieval.py, llm_lib_evolution.py

1. Add function `call_llm_and_parse_with_retry()` in utils.py:

- Wrap the logic of calling the LLM and retrying on failure into a reusable function for better robustness and reusability.

- Encapsulate the LLM APIs of different providers into a unified function, enabling automatic detection and invocation.

2. Add multithread for `propose_variants()` in llm_lib_evolution.py.


### [Update] src/experience_library.py

Add features about library insight management.




------------
# 2025/07/04
------------
## Completed

### [Add] training_data_inspection.ipynb

For first 200 tasks in OR-Instruct, try:

1. Mark the data that can be solved stably and correctly using the base model. -> Filter them when run library learning.

related data:
(data\optimization_tasks\train\tagged\orinstruct_gpt-4o-mini_0_200.json)

2. Check whether the task description aligns the model and code.

related data:
(failed_task_check_orinstruct_gemini-2.5-pro.json)

### [Add] src/llm_lib_retrieval.py

Implement LLM agent that retrieves insights by conditions.

related data:
(task_matched_insights.json)

### [Updated] iterative_learning_and_evaluation.py (previous: iterative_learning.py)

Implement algo 1 in overleaf.

### [Updated] src/llm_opt_diagnoser.py

Add new features. Organize code and fix bugs. Edit and refine prompts.


### [Updated] src/experience_library.py, src\llm_opt_generator.py, src\utils.py

Add new features. Organize code and fix bugs.

## Ongoing 

llm_lib_evolution.py -> features in algo 2 in overleaf
iterative_learning_and_evaluation.py -> pipeline for library evolution




------------
# 2025/06/27
------------
## Completed

1. Updated the Insight object data structure (including fields such as id, condition, explanation, example, version, etc.)

2. Improved core features in class ProgramGenerator and class ProgramDiagnoser (e.g., refining previous insights)

3. Added data management features in the  ExperienceLibrary (retrieval, addition, update, and saving)

4. complete an initial pipeline for training loop in function `run_iterative_learning()`