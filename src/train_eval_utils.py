import os
import re
import time
import json

import traceback
import subprocess
from typing import List, Tuple, Optional, Any, Callable

from src.utils import cal_time_cost
from src.dataloader import DataLoader, Task          
from src.experience_library import ExperienceLibrary
from src.llm_retriever import LibraryRetrieval
from .utils import call_llm_and_parse_with_retry

#* Configure
from omegaconf import OmegaConf
config = OmegaConf.load("train_config.yaml")

def divide_insight(insight):
    # Divide insights into formulation and program stage
    formulation_ins = [
        ins for ins in insight
        if any(k in (ins.get("taxonomy") or {}) for k in ("Domain Modeling", "General Formulation"))
    ]
    program_ins = [ins for ins in insight if "Code Implementation" in ins.get("taxonomy", {})]

    return formulation_ins, program_ins


def generate_solution_with_retrieval(
            iter, task, library, llm_retri, llm_opt, 
            retrieved_insights=[],
            output_path="", verbose=False, save_data=True
            ):
    """
    Retrieve formulation insights -> Generate formulation -> Retrieve program insights -> Generate program
    Returns:
        candidate_formulation, program_output, runnable, is_time_out, retrieved_ins_ids
    """
    formulation_ins, program_ins = divide_insight(retrieved_insights)

    if not formulation_ins and any(key in ins.taxonomy for ins in library for key in ("General Formulation", "Domain Modeling")):
        # print(f"Retrieving formulation insights for Task {task.id}!")
        formulation_ins = llm_retri.retrieve_applicable_insights(
            iter=iter, task=task, stage="Formulation", config=config,
            verbose=verbose, save_data=save_data, output_path=output_path
        )
        # print(f"Retrieved {len(formulation_ins)} formulation insights for Task {task.id}!")

    # Generate mathematical formulation
    candidate_formulation = llm_opt.generate_formulation(
        iter=iter, task=task, retrieved_insights=formulation_ins, abl_params=config.ablation,
        verbose=verbose, save_data=save_data, output_path=output_path
    )

    if not candidate_formulation:
        return None, None, None, None, None, None

    # Retrieve insights for program generation
    if not program_ins and any("Code Implementation" in ins.taxonomy for ins in library):
        # print(f"Retrieving program insights for Task {task.id}!")
        program_ins = llm_retri.retrieve_applicable_insights(
            iter=iter, task=task, stage="Program", formulation=candidate_formulation, config=config,
            verbose=verbose, save_data=save_data, output_path=output_path
        )
        # print(f"Retrieved {len(program_ins)} program insights for Task {task.id}!")

    # Generate solver program
    candidate_program, output, runnable, is_time_out = llm_opt.generate_program(
        iter=iter, task=task, retrieved_insights=program_ins, formulation=candidate_formulation, abl_params=config.ablation,
        verbose=verbose, save_data=save_data, output_path=output_path
    )

    prev_insights = formulation_ins + program_ins

    return prev_insights, candidate_formulation, candidate_program, output, runnable, is_time_out


def is_optimal_with_tolerance(output, gt, tol=config.params.tolerance, mode="absolute"):

    if mode == "absolute":
        if abs(output - gt) <= tol:
            return True
        else:
            return False
    if mode == "relative":
        if abs(output - gt) <= tol * abs(gt):
            return True
        else:
            return False


def check_optimality(task, output, runnable, is_time_out):
    """
    Check if the output is optimal, non-optimal, or a failure to solve/run
    Returns
    -------
    (is_optimal: bool, status: str, feedback: str)
        - is_optimal : True iff output is numeric and within tolerance of ground_truth
        - status     : one of {"optimal", "not_optimal", "failure_solve", "solver_time_out", "run_error"}
        - feedback   : hints for code correction or debugging
    """

    if isinstance(output, float):
        if is_optimal_with_tolerance(output=output, gt=task.ground_truth):
            return True, "optimal", None
        else:
            # Non-optimal results
            feedback = f"\n   [Task {task.id}]: Output was not optimal: {output}. Expected optimal value: {task.ground_truth}"
            return False, "not_optimal", feedback

    # No numeric objective returned
    if runnable:    
        if is_time_out:
            feedback = f"\n   [Task {task.id}]: Solver timed out without finding an optimal solution: \n{output}"
            return False, "solver_time_out", feedback

        feedback = f"\n   [Task {task.id}]: Failed to obtain an objective value: \n{output}"
        return False, "failure_solve", feedback

    # Program not runnable
    feedback = f"\n   [Task {task.id}]: Failed to generate a runnable program: \n{output}"
    return False, "run_error", feedback


def self_verify_test(iter, task, llm_opt, new_insights, prev_insights, save_data=False, output_path=""):
    # Combine new and previous insights
    prev_formulation_ins, prev_program_ins = divide_insight(prev_insights)
    new_formulation_ins, new_program_ins = divide_insight(new_insights)

    all_formulation_ins = prev_formulation_ins + new_formulation_ins
    all_program_ins = prev_program_ins + new_program_ins

    #* Call back and verify the effectiveness of relevant insights to the task
    candidate_formulation = llm_opt.generate_formulation(
        iter=iter,
        task=task,
        retrieved_insights=all_formulation_ins,
        abl_params=config.ablation,
        verbose=False,
        save_data=save_data,
        output_path=os.path.join(output_path, "self_verify")
    )

    _, output, _, _ = llm_opt.generate_program(
        iter=iter,
        task=task,
        retrieved_insights=all_program_ins,
        formulation=candidate_formulation,
        abl_params=config.ablation,
        verbose=False,
        save_data=save_data,
        output_path=os.path.join(output_path, "self_verify")
    )

    # Check optimality
    if isinstance(output, float) and is_optimal_with_tolerance(output=output, gt=task.ground_truth):
        return True
    else:
        return False


def save_checkpoint(library, tasks, metrics, paths, suffix):
    if library:
        # Save latest library and updated taxonomy
        library.save(f"{paths.lib_dir}/library_{suffix}.json")
        library.save_taxonomy(f"{paths.lib_dir}/latest_taxonomy_{suffix}.json")
    # Save tasks with status record
    if tasks:
        tasks.save_as_json(f"{paths.train_output_dir}/train_tasks_record_{suffix}.json")
    if metrics:
        # Save iteration metrics log
        with open(paths.metrics_log_path, "w") as f:
            json.dump(metrics, f, indent=2)


def extract_code(text: str) -> str:
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


def execute_code(code_str, timeout_sec=400):
    try:
        # Using subprocess to execute the code as a separate process
        result = subprocess.run(
            ["python", "-u", "-"], 
            input=code_str,
            text=True, 
            capture_output=True, 
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
    

def self_debug(
    task: "Task" = None,
    failed_program: str = None,
    feedback: str = None,
    config: str = None
) -> Tuple[bool, Optional[str]]:           
    """
    Self-debug the failed program with LLM
    """
    runnable = False                    
    current_program  = failed_program
    current_feedback = feedback

    for attempt in range(1, config.ablation.max_debug_retry + 1):

        # Construct the prompt for diagnosis
        prompt = PROMPT_SELF_DEBUG.format(
            failed_program      = current_program,
            feedback            = current_feedback        
        )
        
        try:
            corrected_program = call_llm_and_parse_with_retry(
                model       = config.model,
                service     = config.service,
                prompt      = prompt,
                # Extract code script from LLM response
                parse_fn    = extract_code,
                temperature = 0,
                max_retry   = 3,                  
                sleep_sec   = 2,
                verbose     = False
            )

            # Update prompt context with new failed program
            current_program  = corrected_program

        except Exception as err:
            print(f"\n   [WARNING] Task {task.id}: Handle malformed LLM outputs after maximum retry as failing to correct program\n")
            traceback.print_exc() # print error and cause
            return False, False

        #* Execute the corrected program
        try:
            output = execute_code(corrected_program) 
            runnable = True
            is_time_out = False
            #* Add solver time limitation to avoid large time cost on solving single task
            if isinstance(output, subprocess.TimeoutExpired):
                is_time_out = True
            else:
                try:
                    output = float(output) # ensure numerical outputs

                except (TypeError, ValueError):
                    pass # keep original output

            # Check optimality when the program is runnable
            is_optimal, _, current_feedback = check_optimality(task=task, output=output, runnable=runnable, is_time_out=is_time_out)
            return is_optimal, runnable

        except Exception as err:
            # Update prompt context with feedback about execution error
            current_feedback = f"Execution error:\n {err.stderr}"

    # Reached maximum retry for correction without successful execution
    is_optimal = False

    return is_optimal, runnable



PROMPT_SELF_DEBUG="""
You are an expert in Industrial Engineering and Operations Research. 

You are given:
1. A Gurobi program failed to execution
2. The execution error message for the failed program


### The failed program
{failed_program}


### Error message
{feedback}


### Your task
Your task is to review the execution error message, identify the issues in the failed program that caused the error, and revise the program so that it can run successfully.


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