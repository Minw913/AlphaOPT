import os
import json
import time
import yaml
from typing import List, Tuple, Optional, Any

from tqdm.auto import tqdm

import concurrent.futures

from src.utils import cal_time_cost
from src.dataloader import DataLoader, Task          
from src.llm_programmer import ProgramGenerator
from src.experience_library import ExperienceLibrary
from src.llm_retriever import LibraryRetrieval
from src.train_eval_utils import check_optimality, self_debug 

def evaluate(
    tasks: List["Task"],
    llm_opt: "ProgramGenerator",
    use_library: bool,
    library: Optional["ExperienceLibrary"],
    config: Any,
) -> Tuple[int, int, int, float]:
    """
    Evaluate the task success rate of a learned experience library on a test dataset
    If use_library is False, the library is not used in the evaluation
    
    Returns:
        n_success: number of successful tasks (pass@1)
        n_runnable: number of runnable tasks
        n_total: total number of tasks
        pass_at_k_rate: pass@k success rate
    """
    n_success = 0
    n_runnable = 0
    # dataset = config.dataset
    output_folder = config.output_folder
    pass_at_k = config.pass_at_k # Default to 1 if not specified

    llm_retri = None

    if use_library:
        llm_retri = LibraryRetrieval(
            lib=library,
            model=llm_opt.model,
            service=config.service,
            temperature=llm_opt.temp,
        )

    def process_task(task, output_dirs):
        """
        Process a single task with multiple attempts for pass@k evaluation
        """
        output_path = f"{output_dirs}/nolib"
        retrieved_ins_ids = []
        formulation_ins, program_ins = [], []

        if use_library:
            # Retrieve relevant insights from an archived experience library
            output_path = f"{output_dirs}/lib"
            formulation_ins = llm_retri.retrieve_applicable_insights(
                    task=task,
                    stage="Formulation",
                    config=config,
                    verbose=False,
                    save_data=True,
                    output_path=output_path
                    )
            retrieved_ins_ids = [ins["insight_id"] for ins in formulation_ins if 'insight_id' in ins]

        # Try multiple times for pass@k evaluation
        attempts_results = []
        for attempt in range(pass_at_k):
            attempt_output_path = f"{output_path}_attempt_{attempt + 1}" if pass_at_k > 1 else output_path
            
            candidate_model = llm_opt.generate_formulation(
                    task=task,
                    retrieved_insights=formulation_ins,
                    # rewrite=bool(config.ablation.rewrite),
                    abl_params=config.ablation,
                    verbose=False,
                    save_data=True,
                    output_path=attempt_output_path
                )
            
            if use_library and config.ablation.include_program_insight:
                program_ins = llm_retri.retrieve_applicable_insights(
                        task=task,
                        stage="Program",
                        config=config,
                        formulation=candidate_model,
                        verbose=False,
                        save_data=True,
                        output_path=attempt_output_path
                    )
                
                retrieved_ins_ids.extend([ins["insight_id"] for ins in program_ins if "insight_id" in ins])

            candidate_program, output, runnable, is_time_out = llm_opt.generate_program(
                    task=task,
                    retrieved_insights=program_ins,
                    formulation=candidate_model,
                    abl_params=config.ablation,
                    verbose=False,
                    save_data=True,
                    output_path=attempt_output_path
                )

            # Check optimality
            is_optimal, status, feedback = check_optimality(task=task, output=output, runnable=runnable, is_time_out=is_time_out)
            
            # Self-Debug
            if config.ablation.max_debug_retry:
                if status == "run_error":
                    is_optimal, runnable = self_debug(task, candidate_program, feedback, config)

            attempts_results.append((int(is_optimal), int(runnable), status))
            
            # If we found a successful solution, we can stop early for pass@k
            if is_optimal:
                break

        # Record task (use the first attempt's results for recording)
        task.retri_ins_lst.append(retrieved_ins_ids)
        task.output_status.append(attempts_results[0][2])  # Use first attempt's status

        # Calculate pass@k results
        pass_at_k_success = any(result[0] for result in attempts_results)  # Any attempt succeeded
        pass_at_k_runnable = any(result[1] for result in attempts_results)  # Any attempt was runnable
        
        return int(attempts_results[0][0]), int(attempts_results[0][1]), int(pass_at_k_success), int(pass_at_k_runnable)
    
    output_dirs = [f"testing/{output_folder}/task_{task.id}" for task in tasks]
    # Use ThreadPoolExecutor to process tasks concurrently
    with concurrent.futures.ThreadPoolExecutor(max_workers=12) as executor:
        results = list(tqdm(executor.map(process_task, tasks, output_dirs), total=len(tasks), desc="Evaluating\n"))

    # Calculate the number of successes and successful executions from the results
    n_success = sum(opt for opt, _, _, _ in results)  # pass@1 success
    n_runnable = sum(run for _, run, _, _ in results)  # pass@1 runnable
    n_pass_at_k_success = sum(pass_k for _, _, pass_k, _ in results)  # pass@k success
    n_pass_at_k_runnable = sum(pass_k_run for _, _, _, pass_k_run in results)  # pass@k runnable
    
    pass_at_k_rate = n_pass_at_k_success / len(tasks) if len(tasks) > 0 else 0.0
    
    return n_success, n_runnable, len(tasks), pass_at_k_rate


def load_config(config_file: str) -> dict:
    """
    Load configuration from a YAML file
    """
    # with open(config_file, "r") as f:
    #     config = yaml.safe_load(f)  
    #* Configure
    from omegaconf import OmegaConf
    config = OmegaConf.load("eval_config.yaml")

    #* Generate a timestamp and append it to output_folder
    # ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    # config.output_folder = f"{config.output_folder}_{ts}"
    # Re-resolve
    # OmegaConf.resolve(config)
    return config


def main() -> None:
    # Read the configuration file path
    config = load_config("./eval_params.yaml")

    # Check if library_path is provided; if not, set use_library flag to False
    # use_library = bool(config.get("library_path", None))
    use_library = bool(config.library_path)

    # Load test tasks
    test_tasks = DataLoader(config.data_path, mode="test")

    if use_library:
        # Load trained experience library
        print("Loading Library...")
        library = ExperienceLibrary.from_json_file(config.library_path)
        test_tasks_save_path = f"./testing/{config.output_folder}/tasks_record_lib.json"
    else:
        print("Do task without Library...")
        library = None
        test_tasks_save_path = f"./testing/{config.output_folder}/tasks_record_nolib.json"

    # Initialize ProgramGenerator
    llm_opt = ProgramGenerator(
        model       = config.model,
        service     = config.service,
        temperature = config.temperature,
    )

    # Run evaluation
    start_time = time.time()
    n_success, n_runnable, n_total, pass_at_k_rate = evaluate(test_tasks, llm_opt, use_library, library, config)
    success_rate = round(n_success / n_total, 3)
    execution_rate = round(n_runnable / n_total, 3)
    pass_at_k = config.pass_at_k

    # Count time cost
    eval_duration = cal_time_cost(start_time, f'Evaluation')

    print(f"\n================  EVALUATION RESULT  ================\n"
            f"Tasks evaluated : {n_total}\n"
            f"Pass@1 Success  : {n_success}\n"
            f"Pass@1 Rate     : {success_rate:.3%}\n"
            f"Pass@{pass_at_k} Rate     : {pass_at_k_rate:.3%}\n"
            f"Execution-rate  : {execution_rate:.3%}\n"
            f"Time cost       : {eval_duration}\n"
            f"====================================================\n")

    # Load existing logs if they exist, otherwise create a new list
    results_path = "./testing/all_test_results.json"
    if os.path.exists(results_path):
        with open(results_path, "r") as f:
            all_results = json.load(f)
    else:
        all_results = []

    # Append current experiment's results to the log list
    all_results.append({
        "dataset":      config.dataset,
        "data_path":    config.data_path,
        "library_path": config.library_path if use_library else "None",
        "model":        config.model,
        "service":      config.service,
        "temperature":  config.temperature,
        "pass_at_k":    pass_at_k,
        "n_total":      n_total,
        "n_success":    n_success,
        "n_runnable":   n_runnable,
        "pass_at_1_rate": success_rate,
        "pass_at_k_rate": pass_at_k_rate,
        "execution_rate": execution_rate,
        "taxonomy":     config.ablation.taxonomy,
        "rewrite":      config.ablation.rewrite,
        "include_example": config.ablation.include_example,
        "include_program_insight": config.ablation.include_program_insight,
        "max_debug_retry": config.ablation.max_debug_retry,
        "duration":     eval_duration,
        "timestamp":    time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime(start_time)),  # Add timestamp
    })

    # Save the updated log
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2)

    # Save tasks with status record
    test_tasks.save_as_json(test_tasks_save_path)


if __name__ == "__main__":
    main()