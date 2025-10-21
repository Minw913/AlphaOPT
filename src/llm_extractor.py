import os
import re
import json
import copy
import subprocess
from itertools import chain
from typing import Optional, List, Tuple
import traceback

from .utils import save_log_data, call_llm_and_parse_with_retry, extract_json_array, extract_json_object
from .dataloader import DataLoader, Task
from .prompts.prompts_ins import PROMPT_INS_FROM_FORMU, PROMPT_INS_FROM_PROGRAM, PROMPT_CONDUCT_MERGE, PROMPT_ONLINE_MERGE

class InsightExtractor:
    """
    LLM_ins agent: Extract labeled insights from code corrections
    """
    def __init__(self, model: str, service: str, temperature: float | None = None):
        self.model = model
        self.service = service
        self.temp = temperature


    def extract_insights(self, text: str) -> dict:
        candidate = None
        try:
            raw = text
            # Extract content between the first '{' and the matching last '}'
            s, e = raw.find('['), raw.rfind(']')
            if s != -1 and e != -1 and e > s:
                candidate = raw[s:e+1]
            else:
                # Grab outermost '{' ... '}' (single object) and wrap later
                s, e = raw.find('{'), raw.rfind('}')
                if s == -1 or e == -1 or e <= s:
                    raise ValueError("No JSON array/object found in LLM output.")
                candidate = raw[s:e+1]

            cand = candidate.strip()

            # Remove trailing commas before ']' or '}'
            cand = re.sub(r",\s*(\]|\})", r"\1", cand)

            # Escape invalid backslashes (i.e., not followed by a valid JSON escape char)
            cand = re.sub(r'(?<!\\)\\(?!["\\/bfnrtu])', r'\\\\', cand)

            # Parse JSON
            result = json.loads(cand)
            # Normalize to list
            if isinstance(result, dict):
                result = [result]
            if not isinstance(result, list):
                raise ValueError(f"Parsed content is not a list; got {type(result).__name__}")
            # Ensure every element is a dict
            for idx, item in enumerate(result):
                if not isinstance(item, dict):
                    raise ValueError(f"Insight at index {idx} is not a dict: {type(item).__name__}")
            
            return result

        except Exception as e:
            # Diagnostic output
            print("LLM raw text:\n", text)
            print("Extracted JSON candidate:\n", candidate if candidate is not None else '<no candidate>')
            print("Error during extract_insights:", repr(e))
            raise


    def generate_insights(
        self, 
        iter: int = None, 
        task: "Task" = None, 
        corrected_program: str = None, 
        failed_formulation: str = None, 
        taxonomy: List[dict] = None,
        verbose: bool = True,
        save_data: bool = False,
        output_path: str = "learning"
        ):
        
        if failed_formulation:
            stage = "Formulation"
            # Extract new insights based on comparison between proposed formulation and gold-standard program
            prompt = PROMPT_INS_FROM_FORMU.format(
                            problem_description=task.desc, 
                            failed_formulation=failed_formulation,
                            correct_program=task.correct_program,
                            domain_taxo=json.dumps(taxonomy["Domain Modeling"], indent=2, ensure_ascii=False),
                            formulation_taxo=json.dumps(taxonomy["General Formulation"], indent=2, ensure_ascii=False),
                            )
        elif corrected_program:
            stage = "Program"
            # Extract new insights based on fixed program
            prompt = PROMPT_INS_FROM_PROGRAM.format(
                            corrected_program=corrected_program,
                            code_taxo=json.dumps(taxonomy["Code Implementation"], indent=2, ensure_ascii=False)
                            )

        custom_header = f"\n==========\n[Iteration {iter}] Generate insights for Task {task.id}\n==========\n"
        error_message = f"\n   Task {task.id} failed to extract generated insights after maximum attempts\n"

        try:
            # Call the LLM and parse the output
            new_insights = call_llm_and_parse_with_retry(
                model=self.model,
                service=self.service,
                prompt=prompt,
                # Extract insights from LLM response
                parse_fn=self.extract_insights, 
                temperature=self.temp,
                max_retry=12,
                sleep_sec=0.5,
                verbose=verbose,
                log_header=custom_header,
                error_message=error_message,
            )

        except Exception as err:
            print(f"\n   [WARNING] Task {task.id} Handle malformed LLM outputs after maximum retry as no insight generated\n")
            traceback.print_exc() # print error and cause
            return []

        # Enrich each insight with default id and task metadata
        for i, ins in enumerate(new_insights):
            new_insights[i] = {
                "insight_id": -1,        
                **ins,                      
                "task_id": task.id,
                "iteration": iter        
            }
        
        if save_data:
            # Save the insights to a JSON file
            insights_path = f"{output_path}/{stage}/extracted_insights_iter_{iter}.json"
            new_insights_copy = copy.deepcopy(new_insights)
            for ins in new_insights_copy:
                taxo = ins.get("taxonomy", {})
                norm = {}
                # lvl1 = stage (e.g., "General Formulation"); lvl1_val = {level1_name: {level2: (null|str)}}
                for lvl1, lvl1_val in taxo.items():
                    if not isinstance(lvl1_val, dict):
                        norm[lvl1] = {}
                        continue
                    # Map each level1_name to the list of all level2 label names under it.
                    norm[lvl1] = {
                        level1_name: (list(level2_dict.keys()) if isinstance(level2_dict, dict) else [])
                        for level1_name, level2_dict in lvl1_val.items()
                    }
                ins["taxonomy"] = norm
            
            save_log_data(new_insights_copy, insights_path)

        return new_insights


    def conduct_insight_merge(
        self, 
        candidate_insights: List[dict] = None, 
        target: int = None,
        verbose: bool = False
        ):
        mapping_ids = {ins["insight_id"]: ins["task_id"] for ins in candidate_insights}
        kept_fields = ["insight_id", "taxonomy", "condition", "explanation", "example"]
        insights_to_be_merge = [{k: d[k] for k in kept_fields if k in d} for d in candidate_insights]

        prompt = PROMPT_CONDUCT_MERGE.format(candidate_insights=json.dumps(insights_to_be_merge, indent=2, ensure_ascii=False))

        custom_header = f"\n==========\nMerge insights in {target}\n==========\n"
        error_message = f"\n   {target} failed to conduct insight merge after maximum attempts\n"

        try:
            # Call the LLM and parse the output
            merge_results = call_llm_and_parse_with_retry(
                model=self.model,
                service=self.service,
                prompt=prompt,
                parse_fn=extract_json_array, 
                temperature=self.temp,
                max_retry=8,
                sleep_sec=0.5,
                verbose=verbose,
                log_header=custom_header,
                error_message=error_message,
            )

        except Exception as err:
            print(f"\n   [WARNING] {target} Handle malformed LLM outputs after maximum retry as no insight merge\n")
            traceback.print_exc() # print error and cause
            return []
        
        
        # Remove 'reason' field ("insight_id": -1)
        merged_insights = [
            {
                **{k: v for k, v in candidate.items() if k != "reason"},
                # flatten the mapping_ids[mid] list and deduplicate
                "task_id": list(set(chain.from_iterable(
                    [mapping_ids[mid]] if isinstance(mapping_ids[mid], str) else mapping_ids[mid]
                    for mid in candidate.get("merged_ids", [])
                    if mid in mapping_ids
                )))
            }
            for candidate in merge_results
        ]

        return merged_insights


    def conduct_insight_online_merge(
        self, 
        new_insight: List[dict] = None, 
        library: "ExperienceLibrary" = None,
        verbose: bool = False
        ):
        
        # Retrieve the existing insights in the library that match the taxonomy of the new insight
        matched_taxo = new_insight[0].get("taxonomy", {})
        existing_insights = library.retrieve_by_taxonomy(query_taxonomy=matched_taxo, include_task_id=True)
        # If no existing insights match the taxonomy, return empty results
        if not existing_insights:
            # print("no existing insights match the taxonomy")
            return [], {}, []
        
        mapping_ids = {ins["insight_id"]: ins["task_id"] for ins in existing_insights}
        # Create mapping from task_id to iteration
        mapping_task_to_iter = {}
        for ins in existing_insights:
            task_id = ins["task_id"]
            iteration = ins["iteration"]
            if isinstance(task_id, list):
                # If task_id is a list, map each task_id to the iteration
                for tid in task_id:
                    mapping_task_to_iter[tid] = iteration
            else:
                # If task_id is a single value, map it directly
                mapping_task_to_iter[task_id] = iteration

        kept_fields = ["taxonomy", "condition", "explanation", "example"]
        new_insight_for_merge = [{k: d[k] for k in kept_fields if k in d} for d in new_insight]
        kept_fields.append("insight_id")
        existing_insights_for_merge = [{k: d[k] for k in kept_fields if k in d} for d in existing_insights]
        
        # Merge the new insight with the existing insights in the library
        prompt = PROMPT_ONLINE_MERGE.format(new_insight=json.dumps(new_insight_for_merge, indent=2, ensure_ascii=False),
                                            existing_insights=json.dumps(existing_insights_for_merge, indent=2, ensure_ascii=False))
        # print("prompt", prompt)
        try:
            # Call the LLM and parse the output
            merge_results = call_llm_and_parse_with_retry(
                model=self.model,
                service=self.service,
                prompt=prompt,
                parse_fn=extract_json_object, 
                temperature=self.temp,
                max_retry=5,
                sleep_sec=0.5,
                verbose=verbose,
            )

            # print("merge_results", merge_results)
        except Exception as err:
            print(f"\n   [WARNING] Online merge: Handle malformed LLM outputs after maximum retry as no insight merge\n")
            traceback.print_exc() # print error and cause
            return [], {}, []
        
        
        # Skip empty merge results (when LLM decides not to merge)
        if not merge_results or not merge_results.get("merged_ids"):
            return [], {}, []
        
        # Remove 'reason' field ("insight_id": -1) and create single merged insight
        new_task_id = new_insight[0]["task_id"]
        if isinstance(new_task_id, list):
            new_task_id_list = new_task_id
        else:
            new_task_id_list = [new_task_id]
        
        merged_insights = {
            **{k: v for k, v in merge_results.items() if k != "reason"},
            # flatten the mapping_ids[mid] list and deduplicate, then add new_insight task_id
            "task_id": list(set(list(chain.from_iterable(
                [mapping_ids[mid]] if isinstance(mapping_ids[mid], str) else mapping_ids[mid]
                for mid in merge_results.get("merged_ids", [])
                if mid in mapping_ids
            )) + new_task_id_list))
        }
        # print("merged_insights", merged_insights)
        # Build iteration mapping table for all task_ids in merged_insights
        merged_task_to_iter = {}
        for task_id in merged_insights.get("task_id", []):
            if task_id in mapping_task_to_iter:
                merged_task_to_iter[task_id] = mapping_task_to_iter[task_id]
            # For new_insight's task_id, use the latest iteration
            elif task_id == new_insight[0]["task_id"]:
                merged_task_to_iter[task_id] = -1 

        return merged_insights, merged_task_to_iter, existing_insights


# Test on a demo
if __name__ == "__main__":
    from tqdm import tqdm
    from concurrent.futures import ThreadPoolExecutor, as_completed
    from threading import Lock
    from experience_library import ExperienceLibrary

    iter = 1
    dataset="integrated_train_no_label"
    library = ExperienceLibrary()

    lock = Lock()               # Lock to safely update shared variables
    temp_lib = []

    def process_task(task, taxo_snapshot):

        # insights_path = f"./learning/{dataset}/task_{task.id}/applicable_insights_iter_{iter}.json"
        program_path  = f"./learning/{dataset}/task_{task.id}/corrected_program_iter_{iter}.py"

        if os.path.exists(program_path):
            with open(program_path, "r", encoding="utf-8") as f:
                corrected_program = f.read()

            output_path = f"./learning/{dataset}/task_{task.id}/labeled_ins/fulltaxo/"
            os.makedirs(output_path, exist_ok=True)

            llm_ins = InsightExtractor(model="gemini-2.5-pro")
            new_insights = llm_ins.generate_insights(
                iter=iter,
                task=task,
                corrected_program=corrected_program,
                taxonomy=taxo_snapshot,
                verbose=True,
                save_data=True,
                output_path=output_path
            )

            return new_insights

    train_dataset_path = f"./learning/{dataset}/train_tasks_record_iter{iter}.json"
    tasks = DataLoader(train_dataset_path, mode="learn", filter_success_num=None, reset=False)

    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = [
            executor.submit(process_task, task, copy.deepcopy(library.taxonomy)) #* Pass a taxonomy snapshot to each task (to avoid concurrent writes)
            for task in tasks
        ]

        for future in tqdm(as_completed(futures), total=len(futures), desc="Processing tasks\n"):
            new_insights = future.result()
            if new_insights:    
                #* Temporarily store new insights in each iteration
                with lock:
                    temp_lib.extend(new_insights)
                #* Update the shared taxonomy snapshot                           
                library.update_taxonomy(new_insights)


    #* Add the new insights into the experience library
    library.add_insights(temp_lib)
    library.save(f"./data/experience_library/iterations/integrated_train_new_label/library_iter{iter}_fulltaxo.json")
    # Save updated taxonomy
    library.save_taxonomy(f"./data/experience_library/iterations/integrated_train_new_label/latest_taxonomy_iter{iter}_fulltaxo.json")