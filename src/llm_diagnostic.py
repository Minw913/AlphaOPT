import os
import re
import json
import subprocess
from typing import Optional, List, Tuple
import traceback
import itertools

from .utils import save_log_data, extract_json_array, call_llm_and_parse_with_retry
from .dataloader import DataLoader, Task
from .llm_programmer import ProgramGenerator
from .llm_retriever import LibraryRetrieval
from .train_eval_utils import check_optimality, is_optimal_with_tolerance

from .prompts.prompts_diag import PROMPT_DIAGNOSE_ISSUES, PROMPT_INS_POS_NEG, PROMPT_VALIDATE_ISSUES, PROMPT_PROGRAM_DIAG


#* Configure
from omegaconf import OmegaConf
config = OmegaConf.load("train_config.yaml")

class ProgramDiagnostic:
    """
    LLM_diag agent: Provide diagnoses and code corrections from failed programs
    """
    def __init__(self, model: str, service: str, temperature: float | None = None):
        self.model = model
        self.service = service
        self.temp = temperature

    
    def extract_code(self, text: str) -> str:
        """
        Extract a clean Python code snippet from the LLM output
        """
        code_block = None
        try:
            raw = text

            # Try to find a Markdown-style Python code block
            m = re.search(r"```python\s*\n([\s\S]*?)\n```", raw)
            if m:
                code_snippet = m.group(1).strip()
                code_block = m.group(0)  # for debugging
            else:
                # If no explicit Python fence, match any fenced code block
                m2 = re.search(r"```(?:\w*\s*)?\n([\s\S]*?)\n```", raw)
                if m2:
                    code_snippet = m2.group(1).strip()
                    code_block = m2.group(0)  # for debugging
                else:
                    # If neither fence is present, raise an error
                    raise ValueError(
                        "No valid code fence found. Expected a ```python``` block or a generic ``` block."
                    )

            return code_snippet

        except Exception as e:
            print("LLM raw text:\n", text)
            print("Extracted code block:\n", code_block if code_block is not None else '<No code block>')
            print("Error during extract_code:", repr(e))
            raise


    def execute_code(self, file_path, timeout_sec=400):
        try:
            # Using subprocess to execute the code as a separate process
            result = subprocess.run(
                ["python", file_path], 
                capture_output=True, 
                text=True, 
                check=True,
                timeout=timeout_sec # Set the maximum run time
            )

            # Extract Gurobi's objVal (optimal objective value) from stdout
            output = result.stdout
            match = re.search(r"Optimal value\s*[:=]\s*([0-9.+-eE]+)", output)

            if match:
                solution = float(match.group(1))
                return solution
            else:
                return output
            
        except subprocess.TimeoutExpired as err:
            return err


    def _diagnose_issues(
        self,
        iter: int = None,
        task: "Task" = None,
        failed_formulation: str = None,
        feedback: str = None,
        verbose: bool = False,
        save_data: bool = False,
        output_path: str = "learning",
    ):           
        """
        Diagnose the issues in the failed formulation
        """

        # Construct the prompt for diagnosis
        prompt = PROMPT_DIAGNOSE_ISSUES.format(
            problem_description=task.desc,
            failed_formulation=failed_formulation,
            feedback=feedback,
            correct_program=task.correct_program,
        )

        # Call the LLM to generate the answer and extract code from string 
        log_header = (f"\n==========\n[Iteration {iter}] Diagnose the issues in Task {task.id}\n==========\n")
        error_message = f"\n   Task {task.id} failed to diagnose issues from LLM after maximum attempts\n"
        
        try:
            diagnosed_issues = call_llm_and_parse_with_retry(
                model       = self.model,
                service     = self.service,
                prompt      = prompt,
                # Extract code script from LLM response
                parse_fn    = extract_json_array,
                temperature = self.temp,
                max_retry   = 5,                  
                sleep_sec   = 2,
                verbose     = verbose,
                log_header  = log_header,
                error_message = error_message
            )

        except Exception as err:
            print(f"\n   [WARNING] Task {task.id}: Handle malformed LLM outputs after maximum retry as failing to diagnose issues\n")
            traceback.print_exc()
            return {}

        if save_data:
            # Save and run corrected code
            issues_diag_path = f"{output_path}/Diagnosis/issues_diagnosis_iter_{iter}.json"
            save_log_data(diagnosed_issues, issues_diag_path)

        return diagnosed_issues


    def _diagnose_pos_neg(
        self,
        iter: int = None,
        task: "Task" = None,
        failed_formulation: str = None,
        diagnosed_issues: List[dict] = [],
        retrieved_insights: List[dict] = [],
        llm_opt: "ProgramGenerator" = None,
        verbose: bool = False,
        save_data: bool = False,
        output_path: str = "learning",
    ):           
        """
        Diagnose the state of retrieved insights (positive or negative)
        """
        formulation_ins, program_ins = retrieved_insights

        # Construct the prompt for diagnosis
        prompt = PROMPT_INS_POS_NEG.format(
            problem_description=task.desc,
            failed_formulation=failed_formulation,
            diagnosed_issues=json.dumps(diagnosed_issues),
            retrieved_insights=formulation_ins,
        )

        # Call the LLM to generate the answer and extract code from string 
        log_header = (f"\n==========\n[Iteration {iter}] Diagnose the failed mathematical formulation for Task {task.id}\n==========\n")
        error_message = f"\n   Task {task.id} failed to diagnose mathematical formulation from LLM after maximum attempts\n"
        
        try:
            insights_diag = call_llm_and_parse_with_retry(
                model       = self.model,
                service     = self.service,
                prompt      = prompt,
                # Extract code script from LLM response
                parse_fn    = extract_json_array,
                temperature = self.temp,
                max_retry   = 5,                  
                sleep_sec   = 2,
                verbose     = verbose,
                log_header  = log_header,
                error_message = error_message,
            )

        except Exception as err:
            print(f"\n   [WARNING] Task {task.id}: Handle malformed LLM outputs after maximum retry as failing to diagnose insights\n")
            traceback.print_exc() # print error and cause
            return []

        if save_data:
            # Save and run corrected code
            insights_diag_path = f"{output_path}/Diagnosis/ins_pos_neg_diagnosis_iter_{iter}.json"
            save_log_data(insights_diag, insights_diag_path)

        # [{"insight_id":1, "state":"positive"}, ...]
        insights_diag = [{"insight_id": ins["insight_id"], "state": ins["state"]} for ins in insights_diag]

        if all(ins.get("state") in ("positive", "invalid") for ins in insights_diag):
            # It is necessary to generate new insights
            is_retrieve_new = True
            pos_formulation_ins = formulation_ins
            new_formulation = failed_formulation
            updated_issues = diagnosed_issues

        else:
            # Exclude the misleading insights and try to generate formulation and program again
            pos_ins_ids = [ins["insight_id"] for ins in insights_diag if ins["state"] == "positive"]
            pos_formulation_ins = [ins for ins in formulation_ins if ins["insight_id"] in pos_ins_ids]

            new_formulation = llm_opt.generate_formulation(
                iter=iter,
                task=task,
                retrieved_insights=pos_formulation_ins,
                # rewrite=False,
                abl_params=config.ablation,
                verbose=False,
                save_data=True,
                output_path=output_path
            )

            _, output, runnable, is_time_out = llm_opt.generate_program(
                iter=iter,
                task=task,
                retrieved_insights=program_ins,
                formulation=new_formulation,
                abl_params=config.ablation,
                verbose=False,
                save_data=True,
                output_path=output_path
            )

            if save_data:
                formu_path = f"{output_path}/Diagnosis/formu1_iter_{iter}.py"
                save_log_data(new_formulation, formu_path)

            # Check optimality
            is_optimal, _, feedback = check_optimality(task=task, output=output, runnable=runnable, is_time_out=is_time_out)
            if is_optimal:
                # It is not necessary to generate new insights
                is_retrieve_new = False 
                updated_issues = None

            else:
                is_retrieve_new = True 

                #* Diagnose the issues in the new formulation again after removing negative insights
                updated_issues = self._diagnose_issues(
                    iter=iter,
                    task=task,
                    failed_formulation=new_formulation,
                    feedback=feedback,
                    verbose=False,
                )

        insights_diag = {ins["insight_id"]: ins["state"] for ins in insights_diag}

        return insights_diag, pos_formulation_ins, is_retrieve_new, new_formulation, updated_issues

    def _diagnose_unretrieved(
        self,
        iter: int = None,
        task: "Task" = None,
        failed_formulation: str = None,
        retrieved_insights: List[dict] = [],
        diagnosed_issues: List[dict] = [],
        llm_opt: "ProgramGenerator" = None,
        llm_retri: "LibraryRetrieval" = None,
        verbose: bool = False,
        save_data: bool = False,
        output_path: str = "learning",            
    ) -> Tuple[bool, Optional[str]]:           
        """
        Diagnose the candidate mathematical formulation
        """
        pos_formulation_ins, program_ins = retrieved_insights

        # {issue_id: applicable_insights}, exclude pos_formulation_ins
        exclude_ids = [ins["insight_id"] for ins in pos_formulation_ins]
        issues_applicable_insights = llm_retri.retrieve_insights_for_diagnosis(
                iter=iter,
                task=task,
                formulation=failed_formulation,
                diagnosed_issues=diagnosed_issues,
                filter_fn=lambda ins: ins.insight_id not in exclude_ids, 
                verbose=verbose,
                save_data=save_data,
                output_path=output_path,
        )

        # candidate_ins_set = [
        #     self._dedup_inner_list_by_id(insights)
        #     for insights in issues_applicable_insights.values()
        #     if insights  # * remove empty insight list for any issue
        # ]
        candidate_ins_set = [
            insights for insights in issues_applicable_insights.values()
            if insights  #* remove empty insight list for any issue
        ]

        is_generate_new = True
        # all_solved = False
        combo_issues_status = []
        combo_unretrieved_ins = []
        combo_corrected_forms = []

        # Only iterate the subset combinations
        max_combo_size = 8
        
        for idx, unretrieved_ins in enumerate(itertools.islice(itertools.product(*candidate_ins_set), max_combo_size)):
            unretrieved_ins = list(unretrieved_ins)
            combo_unretrieved_ins.append(unretrieved_ins)
        
        # for idx, unretrieved_ins in enumerate(
        #     itertools.islice(self.unique_combo_gen(candidate_ins_set), max_combo_size)):

            formulation_ins = pos_formulation_ins + unretrieved_ins
            # Generate new formulation
            corrected_formulation = llm_opt.generate_formulation(
                iter=iter,
                task=task,
                retrieved_insights=formulation_ins,
                # rewrite=False,
                abl_params=config.ablation,
                verbose=False,
                save_data=True,
                output_path=output_path
            )
            combo_corrected_forms.append(corrected_formulation)

            # Construct the prompt for diagnosis
            prompt = PROMPT_VALIDATE_ISSUES.format(
                problem_description=task.desc,
                failed_formulation=failed_formulation,
                diagnosed_issues=json.dumps(diagnosed_issues),
                new_formulation=corrected_formulation
            )

            # Call the LLM to generate the answer and extract code from string 
            log_header = (f"\n==========\n[Iteration {iter}] Validate the regenerated mathematical formulation based on NO.{idx+1} unretrieved insights set for Task {task.id}\n==========\n")
            error_message = f"\n   Task {task.id} failed to validate regenerated mathematical formulation from LLM after maximum attempts\n"
            try:
                issues_status = call_llm_and_parse_with_retry(
                    model       = self.model,
                    service     = self.service,
                    prompt      = prompt,
                    # Extract code script from LLM response
                    parse_fn    = extract_json_array,
                    temperature = self.temp,
                    max_retry   = 5,                  
                    sleep_sec   = 2,
                    verbose     = verbose,
                    log_header  = log_header,
                    error_message = error_message
                )
                
            except Exception as err:
                print(f"\n   [WARNING] Task {task.id}: Handle malformed LLM outputs after maximum retry as failing to validate regenerated mathematical formulation\n")
                traceback.print_exc() # print error and cause
                issues_status = []

            # Save the issues status for each combination
            combo_issues_status.append(issues_status)

            #* If all issues are solved, try to generate program and check optimality 
            all_solved = all(item["status"] == "solved" for item in issues_status)
            if all_solved:
                # {insight_id: unretrieved}
                insights_diag = {item["insight_id"]: "unretrieved" for item in unretrieved_ins}
                _, output, _, _ = llm_opt.generate_program(
                    iter=iter,
                    task=task,
                    retrieved_insights=program_ins,
                    formulation=corrected_formulation,
                    abl_params=config.ablation,
                    verbose=False,
                    save_data=True,
                    output_path=output_path
                )

                # Check optimality
                if isinstance(output, (float, int)) and is_optimal_with_tolerance(output=output, gt=task.ground_truth):
                    # It is not necessary to generate new insights
                    is_generate_new = False 
                    new_formulation = None
                    break
                # All solved but not optimal
                else:
                    # is_generate_new = True
                    new_formulation = corrected_formulation
                    break
        
        if (not all_solved) and is_generate_new and combo_issues_status:
            # Count the number of "unsolved" items within each combo
            unsolved_counts = [sum(1 for issue in combo if issue["status"] == "unsolved") for combo in combo_issues_status]
            if unsolved_counts:
                min_count = min(unsolved_counts)
                target_idx = unsolved_counts.index(min_count)  # Choose combination with least unsolved issues
            else:
                target_idx = 0
            
            unretrieved_ins = combo_unretrieved_ins[target_idx]
            new_formulation = combo_corrected_forms[target_idx]
            issues_status = combo_issues_status[target_idx]

            # insights_diag = {
            #     unretrieved_ins[idx]["insight_id"]: "unretrieved"
            #     for idx, issue in enumerate(issues_status)
            #     if issue["status"] == "solved"
            # }

            insights_diag = {
                ins["insight_id"]: "unretrieved"
                for ins, st in zip(unretrieved_ins, issues_status)
                if isinstance(st, dict) and st.get("status") == "solved"
            }

        if save_data:
            formu_path = f"{output_path}/Diagnosis/formu2_iter_{iter}.py"
            issues_path = f"{output_path}/Diagnosis/issues_status_iter_{iter}.json"
            save_log_data(new_formulation, formu_path)
            save_log_data(issues_status, issues_path)
            
        return insights_diag, unretrieved_ins, is_generate_new, new_formulation


    def diagnose_formulation(
        self,
        iter: int = None,
        task: "Task" = None,
        feedback: str = None,
        failed_formulation: str = None,
        retrieved_insights: List[dict] = [],
        llm_opt: "ProgramGenerator" = None,
        llm_retri: "LibraryRetrieval" = None,
        verbose: bool = False,
        save_data: bool = False,
        output_path: str = "learning",
    ) -> Tuple[bool, Optional[str]]:           
        """
        Diagnose failed formulation and the effectiveness of retrieved insights
        """

        #* Step 1: Diagnose the issues in the failed formulation
        diagnosed_issues = self._diagnose_issues(
            iter=iter,
            task=task,
            failed_formulation=failed_formulation,
            feedback=feedback,
            verbose=False, #verbose,
            save_data=save_data,
            output_path=output_path
        )

        #* Step 2: Diagnose the state of retrieved insights (positive or negative)
        insights_diag, pos_formulation_ins, is_retrieve_new, corrected_formulation, updated_issues = self._diagnose_pos_neg(
            iter=iter,
            task=task,
            failed_formulation=failed_formulation,
            diagnosed_issues=diagnosed_issues,
            retrieved_insights=retrieved_insights,
            llm_opt=llm_opt,
            verbose=True,
            save_data=save_data,
            output_path=output_path
        )

        if not is_retrieve_new:
            print("The retrieved insights are sufficient to solve the task after removing negative insights!")
            is_generate_new = False
            return insights_diag, is_generate_new, None, None 

        else:
            retrieved_insights[0] = pos_formulation_ins # Only keep positive insights
            #* Step 3: Diagnose the unretrieved insights
            insights_diag_new, unretrieved_ins, is_generate_new, new_formulation = self._diagnose_unretrieved(
                iter=iter,
                task=task,
                failed_formulation=corrected_formulation,
                retrieved_insights=retrieved_insights,
                diagnosed_issues=updated_issues,
                llm_opt=llm_opt,
                llm_retri=llm_retri,
                verbose=False,
                save_data=save_data,
                output_path=output_path,            
            )  

            if not is_generate_new:
                print("The existing insights are sufficient to solve the task after adding unretrieved insights!")

            insights_diag.update(insights_diag_new)
            
            updated_formulation_ins = pos_formulation_ins + unretrieved_ins

            return insights_diag, updated_formulation_ins, is_generate_new, new_formulation


    def diagnose_program(
        self,
        iter: int = None,
        task: "Task" = None,
        failed_program: str = None,
        feedback: str = None,
        verbose: bool = False,
        save_data: bool = False,
        output_path: str = "learning",
    ) -> Tuple[bool, Optional[str]]:           
        """
        Diagnose and correct the failed program with LLM
        """
        max_retry_correct = 12  
        runnable = False                    
        current_program  = failed_program
        current_feedback = feedback

        for attempt in range(1, max_retry_correct + 1):

            # Construct the prompt for diagnosis
            prompt = PROMPT_PROGRAM_DIAG.format(
                failed_program      = current_program,
                feedback            = current_feedback,
                # correct_program     = task.correct_program,
            )

            # Call the LLM to generate the answer and extract code from string 
            log_header = (f"\n==========\n[Iteration {iter}] Diagnose and correct the failed program for Task {task.id} at attempt {attempt} \n==========\n")
            error_message = f"\n   Task {task.id} failed to extract code from LLM after maximum attempts\n"
            
            try:
                corrected_program = call_llm_and_parse_with_retry(
                    model       = self.model,
                    service     = self.service,
                    prompt      = prompt,
                    # Extract code script from LLM response
                    parse_fn    = self.extract_code,
                    temperature = self.temp,
                    max_retry   = 3,                  
                    sleep_sec   = 2,
                    verbose     = verbose,
                    log_header  = log_header,
                    error_message = error_message
                )

                # Update prompt context with new failed program
                current_program  = corrected_program

            except Exception as err:
                print(f"\n   [WARNING] Task {task.id}: Handle malformed LLM outputs after maximum retry as failing to correct program\n")
                traceback.print_exc() # print error and cause
                return False, None, None, None

            # Save and run corrected code 
            program_path = (f"{output_path}/corrected_program_iter_{iter}.py")
            save_log_data(corrected_program, program_path)

            #* Execute the corrected program
            try:
                output = self.execute_code(program_path)
                runnable = True
                is_time_out = False
                #* Add solver time limitation to avoid large time cost on solving single task
                if isinstance(output, subprocess.TimeoutExpired):
                    print(f"\n   [Task {task.id}] exceeded maximum run time and was terminated\n")
                    is_time_out = True
                else:
                    try:
                        output = float(output) # ensure numerical outputs

                    except (TypeError, ValueError):
                        pass # keep original output

                # Check optimality when the program is runnable
                is_optimal, output_status, current_feedback = check_optimality(task=task, output=output, runnable=runnable, is_time_out=is_time_out)
                # if runnable, output the current feedback
                return is_optimal, runnable, corrected_program, current_feedback

            except Exception as err:
                # Update prompt context with feedback about execution error
                current_feedback = f"Execution error:\n {err.stderr}"
                print(f"\n   [Task {task.id}] failed to execute program on attempt {attempt}:\n{err.stderr}.")

        # Reached maximum retry for correction without successful execution
        print(f"\n   [Task {task.id}]: Maximum retry reached. Failed to fix the program. Skip!")
        corrected_program = None
        is_optimal = None
        current_feedback = None

        return is_optimal, runnable, corrected_program, current_feedback


# Test on a demo
if __name__ == "__main__":
    from tqdm import tqdm
    from concurrent.futures import ThreadPoolExecutor, as_completed
    from threading import Lock
    from experience_library import ExperienceLibrary