import json
import os
import re
from typing import Optional, Callable, List, Any
import traceback

from .experience_library import ExperienceLibrary, Insight
from .utils import save_log_data, call_llm_and_parse_with_retry, extract_json_array
from .dataloader import DataLoader, Task
from .prompts.prompts_diag import PROMPT_RETRI_LABEL, PROMPT_RETRI_INS
from .prompts.prompts_retri import PROMPT_QUICK_MATCH_MODEL, PROMPT_FULL_CHECK_MODEL, PROMPT_QUICK_MATCH_CODE, PROMPT_FULL_CHECK_CODE


class LibraryRetrieval:
    """
    LLM_retri agent: Retrieve applicable insights based on taxonomy
    """
    def __init__(self, lib: "ExperienceLibrary", model: str, service: str, temperature: float | None = None):
        self.library = lib     # use an ExperienceLibrary instance
        self.model = model
        self.service = service
        self.temp = temperature


    def extract_taxonomy(self, text: str):
        """
        Extract the first JSON *object* from an LLM output and return it as a Python dict
        """
        candidate = None
        try:
            # Keep original for debugging
            raw = text

            # Locate the outermost JSON object
            start = raw.find('{')
            end   = raw.rfind('}')
            if start == -1 or end == -1 or end <= start:
                raise ValueError("No JSON object found in the text.")
            candidate = raw[start:end+1]

            cand = candidate.strip()

            # Remove trailing commas before ']' or '}'
            cand = re.sub(r",\s*(\]|\})", r"\1", cand)

            # Escape invalid backslashes (i.e., not followed by a valid JSON escape char)
            cand = re.sub(r'\\(?!["\\/bfnrtu])', r'\\\\', cand)

            # Parse JSON
            result = json.loads(cand)
            if not isinstance(result, dict):
                raise ValueError(f"The parsed JSON is not an object (dict); got {type(result).__name__}")
            return result

        except Exception as e:
            print("LLM raw text:\n", text)
            print("Extracted JSON candidate:\n", candidate if candidate is not None else '<no candidate>')
            print("Error during extract_taxonomy:", repr(e))
            raise


    def _process_insights_in_batches(self, 
                    matched_insights: List[dict], 
                    prompt_template: str,
                    prompt_kwargs: dict,
                    batch_size: int = 30,
                    custom_header: str = "",
                    error_message: str = "",
                    verbose: bool = False) -> List[dict]:
        """
        Generic function for processing insights in batches
        Returns:
            Merged list of applicable insight results
        """
        if not matched_insights:
            return []
        
        # Split insights into batches
        batches = [matched_insights[i:i + batch_size] for i in range(0, len(matched_insights), batch_size)]
        all_applicable_results = []
        
        for batch_idx, batch in enumerate(batches):
            # if verbose:
            # print(f"\n   Processing batch {batch_idx + 1}/{len(batches)} insights (total {len(batch)} items)")
            
            # Build prompt for current batch
            batch_prompt_kwargs = prompt_kwargs.copy()
            batch_prompt_kwargs['candidate_insights'] = json.dumps(batch)
            batch_prompt = prompt_template.format(**batch_prompt_kwargs)
            
            # Build custom header for current batch
            batch_header = f"{custom_header}\n[Batch {batch_idx + 1}/{len(batches)}]"
            batch_error = f"{error_message} (Batch {batch_idx + 1})"
            
            try:
                # Call LLM to process current batch
                batch_results = call_llm_and_parse_with_retry(
                    model=self.model,
                    service=self.service,
                    prompt=batch_prompt,
                    parse_fn=extract_json_array,
                    temperature=0,
                    max_retry=5,
                    sleep_sec=0.5,
                    verbose=verbose,
                    log_header=batch_header,
                    error_message=batch_error
                )
                
                # Merge results
                if batch_results and isinstance(batch_results, list):
                    all_applicable_results.extend(batch_results)
                    
            except Exception as e:
                print(f"\n   [WARNING] Batch {batch_idx + 1}: Failed to parse LLM output, using candidate insights as fallback\n")
                traceback.print_exc()
                # If parsing fails, use current batch insights as fallback
                all_applicable_results.extend(batch)
        
        return all_applicable_results

    def quick_match_by_taxonomy(self,  
                        iter: int = None, 
                        task: "Task" = None,
                        stage: str = None,
                        formulation: str = None,
                        verbose: bool = False,
                        output_path: str = None
                        ):
        """
        Quickly retrieves insights whose taxonomy best match the current task
        Returns the parsed JSON (list of matched insights)
        """
        if stage == "Formulation":
            prompt = PROMPT_QUICK_MATCH_MODEL.format(
                    problem_description=task.desc,
                    domain_taxo=json.dumps(self.library.taxonomy["Domain Modeling"], indent=2, ensure_ascii=False),
                    formu_taxo=json.dumps(self.library.taxonomy["General Formulation"], indent=2, ensure_ascii=False)
                    )

        if stage == "Program":
            prompt = PROMPT_QUICK_MATCH_CODE.format(
                        problem_description=task.desc,
                        mathematical_model=formulation,
                        taxo=json.dumps(self.library.taxonomy["Code Implementation"], indent=2, ensure_ascii=False)
                            )
            
        custom_header = f"\n==========\n[Iteration {iter}] Quickly match library insights by taxnomoy for [{stage}] generation of Task {task.id}\n==========\n"
        error_message = f"\n   [{stage}] Task {task.id} failed to extract matched insight labels after maximum attempts\n"

        try:
            # Call the LLM and parse the output (==== ... ==== JSON block)
            matched_taxo = call_llm_and_parse_with_retry(
                model=self.model,
                service=self.service,
                prompt=prompt,
                parse_fn=self.extract_taxonomy, 
                temperature=0,
                max_retry=5,
                sleep_sec=5,
                verbose=verbose,
                log_header=custom_header,
                error_message=error_message
            )

        # Handle malformed LLM outputs by treating them as no matched taxonomy labels to ensure continued execution
        except Exception as err:
            print(f"\n   [{stage}] Task {task.id}: Handle malformed LLM outputs after maximum retry as no matched taxonomy labels\n")
            traceback.print_exc() # print error and cause
            return {}

        # If no taxonomy labels are matched by LLM
        if matched_taxo == {}: 
            print(f"\n   [{stage}] Task {task.id}: No matched taxonomy labels found in the library\n")
            return {}
        
        # Retrieve the insights under the matched taxonomy labels
        matched_insights = self.library.retrieve_by_taxonomy(query_taxonomy=matched_taxo)

        if output_path:
            # Save the matched taxonomy
            taxo_path = f"{output_path}/{stage}/matched_taxo_iter_{iter}.json"
            save_log_data(matched_taxo, taxo_path)

            # Save the task with its matched insights
            task_matched_insights = {
            "task_id": task.id,
            "description": task.desc,
            "matched_insights": matched_insights
            }
            insights_path = f"{output_path}/{stage}/matched_insights_iter_{iter}.json"
            save_log_data(task_matched_insights, insights_path)

        return matched_insights


    def retrieve_applicable_insights(self,
                iter: int = None,
                task: "Task" = None,
                stage: str = None,
                formulation: str = None,
                filter_fn: Optional[Callable[["Insight"], bool]] = None,  # We can set filter=lambda ins: ins.confidence > 0.7
                batch_size: int = 30,  # umber of insights to process per batch
                config: Optional[Any] = None,
                verbose: bool = False,
                save_data: bool = False,
                output_path: str = "learning"
                ):
        
        if config.ablation.taxonomy:  # Enable taxonomy
            matched_insights = self.quick_match_by_taxonomy(iter=iter, task=task, stage=stage, formulation=formulation, verbose=verbose, output_path=output_path)
            if not matched_insights:
                if verbose:
                    print(f"\n   Task {task.id} : No candidates on [{stage}], skip!\n")
                return []
        else:
            # When taxonomy is disabled, use all insights from the library
            matched_insights = self.library.to_json()
            if not matched_insights:
                if verbose:
                    print(f"\n   Task {task.id} : No insights in library, skip!\n")
                return []

        if stage == "Formulation": 
            prompt_template = PROMPT_FULL_CHECK_MODEL
            prompt_kwargs = {
                "problem_description": task.desc
            }

        elif stage == "Program":
            prompt_template = PROMPT_FULL_CHECK_CODE
            prompt_kwargs = {
                "problem_description": task.desc,
                "mathematical_model": formulation
            }

        custom_header = f"\n==========\n[Iteration {iter}] Check the applicability of insights on [{stage}] for Task {task.id}\n==========\n"
        error_message = f"\n   Task {task.id} [{stage}]: failed to extract applicable insights after maximum attempts\n"

        # Use batch processing function
        applicable_results = self._process_insights_in_batches(
            matched_insights=matched_insights,
            prompt_template=prompt_template,
            prompt_kwargs=prompt_kwargs,
            batch_size=batch_size,
            custom_header=custom_header,
            error_message=error_message,
            verbose=verbose
        )
        # If no insights are matched by LLM
        if applicable_results == []: 
            print(f"\n   Task {task.id} [{stage}]: No applicable insights found in the library\n")

        # Retrieve the insight list with full context for the applicable IDs
        applicable_ids = [ins['insight_id'] for ins in applicable_results if 'insight_id' in ins]
        applicable_insights = self.library.retrieve_insights_by_id(applicable_ids, filter_fn=filter_fn)

        # Save the task with its applicable insights to a json file
        if save_data:
            reason_map = {ins["insight_id"]: ins.get("reason", "") for ins in applicable_results if 'insight_id' in ins}
            applicable_insights_info = [
                {**ins, "reason": reason_map.get(ins["insight_id"])} 
                for ins in applicable_insights
            ]
            task_applicable_insights = {
            "task_id": task.id,
            "description": task.desc,
            "applicable_insights": applicable_insights_info
            }
            insights_path = f"{output_path}/{stage}/applicable_insights_iter_{iter}.json"
            save_log_data(task_applicable_insights, insights_path)

        return applicable_insights


    def retrieve_insights_for_diagnosis(self, 
        iter: int = None,
        task: "Task" = None,
        formulation: str = None,
        diagnosed_issues: List[dict] = [],
        filter_fn: Optional[Callable[[Any], bool]] = None,
        batch_size: int = 30,
        verbose: bool = False,
        save_data: bool = False,
        output_path: str = "learning"
        ):
        
        #* For each diagnosed issue, retrieve applicable insights
        issues_applicable_insights = {}  # {issue_id: applicable_insights}
        for issue in diagnosed_issues:
            idx = issue.get("id")
            #* Match by taxonomy
            prompt = PROMPT_RETRI_LABEL.format(
                problem_description=task.desc, 
                failed_formulation=formulation,
                one_diagnosed_issue=json.dumps(issue),
                domain_taxo=json.dumps(self.library.taxonomy["Domain Modeling"], indent=2, ensure_ascii=False),
                formu_taxo=json.dumps(self.library.taxonomy["General Formulation"], indent=2, ensure_ascii=False)
            )

            custom_header = f"\n==========\n[Iteration {iter}] [Unretrieved Insight] Quickly match library insights by taxnomoy for Task {task.id}\n==========\n for Task {task.id}\n==========\n"
            error_message = f"\n   Task {task.id} failed to extract labels for unretrieved insights after maximum attempts\n"

            try:
                # Call the LLM and parse the output (==== ... ==== JSON block)
                issue_matched_taxo = call_llm_and_parse_with_retry(
                    model=self.model,
                    service=self.service,
                    prompt=prompt,
                    # Output a dict with taxonomy labels or an empty dict {}
                    parse_fn=self.extract_taxonomy, 
                    temperature=0,
                    max_retry=5,
                    sleep_sec=5,
                    verbose=verbose,
                    log_header=custom_header,
                    error_message=error_message
                )

            # Handle malformed LLM outputs by treating them as no matched taxonomy labels to ensure continued execution
            except Exception as err:
                print(f"\n    Task {task.id} [Diagnosis]: Handle malformed LLM outputs after maximum retry as no matched taxonomy labels\n")
                traceback.print_exc() # print error and cause
                return {}

            # If no taxonomy labels are matched by LLM
            if issue_matched_taxo == {}: 
                print(f"\n    Task {task.id} [Diagnosis]: No matched taxonomy labels found in the library\n")
                return {}
        
            # Retrieve the insights under the matched taxonomy labels, exclude already existing insights
            matched_insights = self.library.retrieve_by_taxonomy(query_taxonomy=issue_matched_taxo, filter_fn=filter_fn)

            # Prepare prompt template and parameters
            prompt_template = PROMPT_RETRI_INS
            prompt_kwargs = {
                "problem_description": task.desc,
                "failed_formulation": formulation,
                "one_diagnosed_issue": json.dumps(issue)
            }
            
            custom_header = f"\n==========\n[Iteration {iter}] [Unretrieved Insight] Check applicability for Task {task.id}\n==========\n"
            error_message = f"\n   Task {task.id} [Diagnosis]: Failed to extract applicable insights after maximum attempts\n"

            # Use batch processing function
            applicable_results = self._process_insights_in_batches(
                matched_insights=matched_insights,
                prompt_template=prompt_template,
                prompt_kwargs=prompt_kwargs,
                batch_size=batch_size,
                custom_header=custom_header,
                error_message=error_message,
                verbose=verbose
            )

            # If no insights are matched by LLM
            if applicable_results == []: 
                print(f"\n   Task {task.id} [Diagnosis]: No applicable insights found in the library\n")

            # Retrieve the insight list with full context for the applicable IDs
            applicable_ids = [ins['insight_id'] for ins in applicable_results if 'insight_id' in ins]
            applicable_insights = self.library.retrieve_insights_by_id(applicable_ids)

            issues_applicable_insights[idx] = applicable_insights

            # Save data temporarily
            if save_data:
                # Save the matched taxonomy
                taxo_path = f"{output_path}/Diagnosis/matched_taxo_iter_{iter}_idx_{idx}.json"
                save_log_data(issue_matched_taxo, taxo_path)

                # Save the task with its matched insights
                task_matched_insights = {
                "task_id": task.id,
                "description": task.desc,
                "matched_insights": matched_insights
                }
                matched_insights_path = f"{output_path}/Diagnosis/matched_insights_iter_{iter}_idx_{idx}.json"
                with open(matched_insights_path, "w") as fout:
                    json.dump(task_matched_insights, fout, indent=2, ensure_ascii=False)
                
                # Save the applicable insights
                reason_map = {
                    ins["insight_id"]: (ins.get("ranking"), ins.get("reason", ""))
                    for ins in applicable_results if 'insight_id' in ins
                }
                applicable_insights_info = [
                    {
                        **ins,
                        "ranking": reason_map.get(ins["insight_id"], (None, None))[0],
                        "reason": reason_map.get(ins["insight_id"], (None, None))[1]
                    }
                    for ins in applicable_insights
                ]
                task_applicable_insights = {
                "task_id": task.id,
                "description": task.desc,
                "applicable_insights": applicable_insights_info
                }
                applicable_insights_path = f"{output_path}/Diagnosis/applicable_insights_iter_{iter}_idx_{idx}.json"
                save_log_data(task_applicable_insights, applicable_insights_path)

        return issues_applicable_insights