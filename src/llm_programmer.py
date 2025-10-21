import os
import re
import json
import subprocess
from typing import Optional, List, Tuple
import traceback
from functools import partial

from .experience_library import Insight, ExperienceLibrary
from .llm_retriever import LibraryRetrieval
from .utils import save_log_data, call_llm_and_parse_with_retry, extract_json_array
from src.train_eval_utils import check_optimality
from .dataloader import DataLoader, Task
from .prompts.prompts_opt import PROMPT_GENERATE_FORMU, PROMPT_GENERATE_PROGRAM, PROMPT_INS_REWRITE, PROMPT_SELF_EXPLORE

class ProgramGenerator:
    """
    LLM_opt agent: Formulate and generate program and solution for NL optimization tasks
    """
    def __init__(self, model: str, service: str, temperature: float | None = None):
        self.model = model
        self.service = service
        self.temp = temperature
    

    def extract_text(self, text: str):
        """
        Extract the content inside triple backtick ```...``` fences from LLM output
        """
        full_content = None
        try:
            raw = text

            # Search for the first fenced block enclosed by triple backticks
            m = re.search(r"```([\s\S]*?)```", raw)
            if m:
                content = m.group(1).strip()
                full_content = m.group(0)  # store full block for debugging
            else:
                # Raise an error if no fenced block is found
                raise ValueError("No valid ```...``` block found.")

            return content

        except Exception as e:
            # Debugging info if extraction fails
            print("LLM raw text:\n", text)
            print("Extracted block:\n", full_content if full_content is not None else '<No fenced block>')
            print("Error during extract_model:", repr(e))
            raise        


    def extract_code(self, text: str, formatted_output: str = None) -> str:
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

            return code_snippet + formatted_output

        except Exception as e:
            print("LLM raw text:\n", text)
            print("Extracted code block:\n", code_block if code_block is not None else '<No code block>')
            print("Error during extract_code:", repr(e))
            raise
        

    def execute_code(self, code_str, timeout_sec=400):
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
        
        
    def rewrite_insights(self, iter, task, retrieved_insights, verbose, save_data, output_path):
        # Construct the prompt for solution generation, if insights are provided, incorporate them into the prompt
        prompt = PROMPT_INS_REWRITE.format(
                            problem_description=task.desc, 
                            retrieved_insights=json.dumps(retrieved_insights, indent=2, ensure_ascii=False))

        # print(prompt)
        
        custom_header = f"\n==========\n[Iteration {iter}]: Insight Rewrite for Task {task.id}\n==========\n"
        error_message = f"\n   Task {task.id} failed to rewrite insights from LLM after maximum attempts\n"

        try:
            # Call the LLM and parse the output
            rewritten_results = call_llm_and_parse_with_retry(
                model=self.model,
                service=self.service,
                prompt=prompt,
                # Extract python code from LLM response
                parse_fn=extract_json_array, 
                temperature=self.temp,
                max_retry=8,
                sleep_sec=0.5,
                verbose=verbose,
                log_header=custom_header,
                error_message=error_message 
            )

        except Exception as err:
            print(f"\n   [WARNING] Task {task.id}: Handle malformed LLM outputs after maximum retry as no rewritten insights\n")
            traceback.print_exc() # print error and cause
            return retrieved_insights

        rewritten_insights = [
            {k: v for k, v in ins.items() if k != "decision"}
            for ins in rewritten_results
            if ins.get("decision") is not None
        ]

        if save_data:    
            model_path = f"{output_path}/rewritten_insights_iter_{iter}.json"
            save_log_data(rewritten_insights, model_path)

        return rewritten_insights


    def generate_formulation(
        self, 
        iter: int = None,
        task: "Task" = None, 
        retrieved_insights: List[dict] = None,
        abl_params: bool = False,
        verbose: bool = False,
        save_data: bool = False,
        output_path: str = "learning"
        ):
        
        #* Add rewrite component 
        if abl_params.rewrite and retrieved_insights:
            fields_to_input = ["insight_id", "condition", "explanation"]
            retrieved_insights = self.rewrite_insights(iter, task, retrieved_insights, verbose, save_data, output_path)
        else:
            fields_to_input = ["insight_id", "explanation"]

        #* Add insight example
        if abl_params.include_example:
            fields_to_input.append("example")

        retrieved_insights = [{k: v for k, v in ins.items() if k in fields_to_input} for ins in retrieved_insights]

        # Construct the prompt for solution generation, if insights are provided, incorporate them into the prompt
        prompt = PROMPT_GENERATE_FORMU.format(
                            problem_description=task.desc, 
                            insights=json.dumps(retrieved_insights, indent=2, ensure_ascii=False))

        # print(prompt)

        custom_header = f"\n==========\n[Iteration {iter}]: Formulation Generation for Task {task.id}\n==========\n"
        error_message = f"\n   Task {task.id} failed to extract formulation from LLM after maximum attempts\n"

        try:
            # Call the LLM and parse the output
            formulation = call_llm_and_parse_with_retry(
                model=self.model,
                service=self.service,
                prompt=prompt,
                # Extract python code from LLM response
                parse_fn=self.extract_text, 
                temperature=self.temp,
                max_retry=8,
                sleep_sec=0.5,
                verbose=verbose,
                log_header=custom_header,
                error_message=error_message 
            )

        except Exception as err:
            print(f"\n   [WARNING] Task {task.id}: Handle malformed LLM outputs after maximum retry as no generated model\n")
            traceback.print_exc() # print error and cause
            return None

        if save_data:    
            # Save the model
            model_path = f"{output_path}/model_iter_{iter}.txt"
            save_log_data(formulation, model_path)

        return formulation

    
    def generate_program(
        self, 
        iter: int = None,
        task: "Task" = None, 
        retrieved_insights: List[dict] = None,
        formulation: str = None, 
        abl_params: bool = False,
        verbose: bool = False,
        save_data: bool = False,
        output_path: str = "learning"
        ):
        # Construct the prompt for solution generation, if insights are provided, incorporate them into the prompt
        fields_to_input = ["insight_id", "explanation"]
        #* Add insight example
        if abl_params.include_example:
            fields_to_input.append("example")

        retrieved_insights = [{k: v for k, v in ins.items() if k in fields_to_input} for ins in retrieved_insights]

        prompt = PROMPT_GENERATE_PROGRAM.format(
                            problem_description=task.desc, 
                            mathematical_model=formulation,
                            insights=json.dumps(retrieved_insights, indent=2, ensure_ascii=False))

        custom_header = f"\n==========\n[Iteration {iter}]: Program Generation for Task {task.id}\n==========\n"
        error_message = f"\n   Task {task.id} failed to extract code from LLM after maximum attempts\n"

        try:
            # Call the LLM and parse the output
            program = call_llm_and_parse_with_retry(
                model=self.model,
                service=self.service,
                prompt=prompt,
                # Extract python code from LLM response
                parse_fn=partial(self.extract_code, formatted_output=formatted_output), 
                temperature=self.temp,
                max_retry=8,
                sleep_sec=0.5,
                verbose=verbose,
                log_header=custom_header,
                error_message=error_message 
            )

        except Exception as err:
            print(f"\n   [WARNING] Task {task.id}: Handle malformed LLM outputs after maximum retry as no generated program\n")
            traceback.print_exc() # print error and cause
            return None, None, None, None
        
        try:
            # Execute the code
            output = self.execute_code(program)
            runable = True
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

        except Exception as err:
            output = getattr(err, "stderr", str(err))
            runable = False
            is_time_out = None

        if save_data:    
            # Save the code and output
            program_path = f"{output_path}/program_iter_{iter}.py"
            output_path = f"{output_path}/output_iter_{iter}.txt"
            save_log_data(program, program_path)
            save_log_data(str(output), output_path)

        return program, output, runable, is_time_out
    

    def self_explore(
        self,
        task: "Task" = None,
        failed_program: str = None,
        feedback: str = None,
        verbose: bool = False,
        save_data: bool = False,
        output_path: str = "learning",
    ) -> Tuple[bool, Optional[str]]:           
        """
        Self explore the gold-standard program by LLM
        """
        max_retry_explore = 5  
        runnable = False                    
        current_program  = failed_program
        current_feedback = feedback
        # Record all the attempts as the input
        all_failed_attempts = [{"Name": f'Attempt 1', "Program": failed_program, "Feedback": feedback}]

        for attempt in range(1, max_retry_explore + 1):

            # Construct the prompt for diagnosis
            prompt = PROMPT_SELF_EXPLORE.format(
                problem_description = task.desc,
                failed_attempts = json.dumps(all_failed_attempts, indent=2, ensure_ascii=False),
                ground_truth = task.ground_truth
            )
            # print(prompt)
            # Call the LLM to generate the answer and extract code from string 
            log_header = (f"\n==========\n Self-explore the gold-standard program for Task {task.id}\n==========\n")
            error_message = f"\n   Task {task.id} failed to extract the gold-standard program after maximum attempts\n"
            
            try:
                corrected_program = call_llm_and_parse_with_retry(
                    model       = self.model, 
                    service     = self.service,
                    prompt      = prompt, 
                    # Extract code script from LLM response
                    parse_fn    = partial(self.extract_code, formatted_output=formatted_output), 
                    temperature = 0.7,
                    max_retry   = 3,                  
                    sleep_sec   = 2,
                    verbose     = verbose, #verbose,
                    log_header  = log_header,
                    error_message = error_message
                )

                # print(corrected_program)
                if not corrected_program:
                    continue

                # Update prompt context with new failed program
                current_program  = corrected_program

            except Exception as err:
                print(f"\n   [WARNING] Task {task.id}: Handle malformed LLM outputs after maximum retry as failing to correct program\n")
                traceback.print_exc() # print error and cause
                return False, None

            #* Execute the corrected program
            try:
                output = self.execute_code(current_program)
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

            except Exception as err:
                output = getattr(err, "stderr", str(err))
                runnable = False
                is_time_out = None

            # Check optimality when the program is runnable
            is_optimal, _, current_feedback = check_optimality(task=task, output=output, runnable=runnable, is_time_out=is_time_out)

            if is_optimal:
                gold_standard_program = corrected_program
                print(f"\n   [Task {task.id}]: found the gold-standard program!")
                return is_optimal, gold_standard_program

            all_failed_attempts.append({"Name": f'Attempt {attempt + 1}', "Program": corrected_program, "Feedback": current_feedback})

        # Reached maximum retry for correction without successful execution
        print(f"\n   [Task {task.id}]: Maximum retry reached. Failed to self-explore the gold-standard program. Skip!")
        gold_standard_program = None
        is_optimal = False

        return is_optimal, gold_standard_program

# Standard footer to append
formatted_output = (
    "\n\nif model.Status == GRB.OPTIMAL:\n"
    "    print(\"Optimal value:\", model.ObjVal)\n"
    "elif model.Status == GRB.INFEASIBLE:\n"
    "    print(\"Model is infeasible.\")\n"
    "elif model.Status == GRB.UNBOUNDED:\n"
    "    print(\"Model is unbounded.\")\n"
    "else:\n"
    "    print(\"Other status:\", model.Status)\n"
)


# Test on a demo
if __name__ == "__main__":
    # Load train dataset
    train_dataset_path = "./data/optimization_tasks/train/orinstruct_data.csv"
    tasks = DataLoader(train_dataset_path, mode="learn")

    # Generate program and output using LLM_opt
    llm_opt = ProgramGenerator(model="gemini-2.5-flash")