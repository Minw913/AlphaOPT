import os
import time
import json

from src.dataloader import DataLoader
from src.utils import cal_time_cost
from src.train_eval_utils import save_checkpoint
from library_online_learning import run_library_online_learning
from library_diagnosis import run_library_diagnosis
from library_refinement import run_library_refinement
from src.experience_library import ExperienceLibrary
from src.llm_programmer import ProgramGenerator
from src.llm_diagnostic import ProgramDiagnostic
from src.llm_extractor import InsightExtractor
from src.llm_retriever import LibraryRetrieval
from src.llm_evolver import LibraryEvolution

def print_metrics_summary(metrics_log):
    """
    Print structured summary of metrics from metrics_log.
    
    Args:
        metrics_log: List of dictionaries containing metrics for each iteration
    """
    print("\n" + "="*80)
    print("METRICS SUMMARY")
    print("="*80)
    
    if not metrics_log:
        print("No metrics available.")
        return
    
    # Print header
    print(f"{'Iter':<4} {'Train Acc':<10} {'Library Size':<8} {'Online Merge':<12} {'Self Verify':<12} {'Diagnose':<10} {'Refinement':<12}")
    print("-" * 80)
    
    # Process each iteration's metrics
    for i, metrics in enumerate(metrics_log):
        # Extract basic metrics
        train_accuracy = metrics.get("train_accuracy", "N/A")
        library_size = metrics.get("library_size", "N/A")
        online_merge_rate = metrics.get("online_merge_rate", "N/A")
        self_verify_rate = metrics.get("self_verify_rate", "N/A")
        
        # Extract diagnosis metrics if available
        diagnose_success_rate = "N/A"
        refinement_rate = "N/A"
        
        # Look for Library Diagnosis stage metrics
        if "stage" in metrics and metrics["stage"] == "Library Diagnosis":
            diagnose_success_rate = metrics.get("diagnose_success_rate", "N/A")
            refinement_rate = metrics.get("refinement_rate", "N/A")
        
        # Format values for display
        def format_value(value):
            if value == "N/A":
                return "N/A"
            elif isinstance(value, (int, float)):
                if isinstance(value, float):
                    return f"{value:.3f}"
                else:
                    return str(value)
            else:
                return str(value)
        
        # Print row
        print(f"{i:<4} {format_value(train_accuracy):<10} {format_value(library_size):<8} "
                f"{format_value(online_merge_rate):<12} {format_value(self_verify_rate):<12} "
                f"{format_value(diagnose_success_rate):<10} {format_value(refinement_rate):<12}")
    
    print("="*80)
    
    # Print detailed breakdown for Library Diagnosis stages
    print("\nDETAILED BREAKDOWN:")
    print("-" * 50)
    
    for i, metrics in enumerate(metrics_log):
        if "stage" in metrics and metrics["stage"] == "Library Diagnosis":
            print(f"\nIteration {i} - Library Diagnosis:")
            print(f"  Diagnose Success Rate: {metrics.get('diagnose_success_rate', 'N/A')}")
            print(f"  Refinement Rate: {metrics.get('refinement_rate', 'N/A')}")
            
            # Print other relevant metrics for this stage
            if "train_accuracy" in metrics:
                print(f"  Train Accuracy: {metrics.get('train_accuracy', 'N/A')}")
            if "library_size" in metrics:
                print(f"  Library Size: {metrics.get('library_size', 'N/A')}")
    
    print("\n" + "="*80)

#* Configure
from omegaconf import OmegaConf
config = OmegaConf.load("train_config.yaml")

#* Generate a timestamp and append it to output_folder
# ts = datetime.now().strftime("%Y%m%d-%H%M%S")
# config.output_folder = f"{config.output_folder}_{ts}"
# Re-resolve
OmegaConf.resolve(config)

# Initialize the LLM agents
# Advanced models use OpenRouter
llm_opt = ProgramGenerator(model=config.base_model, service=config.base_service, temperature=0)
llm_diag = ProgramDiagnostic(model=config.advanced_model, service=config.advanced_service, temperature=0)
llm_ins = InsightExtractor(model=config.advanced_model, service=config.advanced_service, temperature=0.7)

temp_online = 0.7 if config.params.max_solution_attempts > 1 else 0
llm_opt_online = ProgramGenerator(model=config.base_model, service=config.base_service, temperature=temp_online)


# 0 (start from online learning), 1 (start from library diagnosis at iter 1)
start_iter = config.start_iter 
end_iter = config.params.num_iterations + 1

if start_iter == 0:
    train_tasks = DataLoader(config.file_paths.train_data_path, mode="learn", filter_success_num=None, reset=True) 
    # Initialize the experience library as an empty list 
    library = ExperienceLibrary()
    # Track iteration metrics
    metrics_log = []  

else:
    if start_iter == 1:
        train_data_path = f"{config.file_paths.train_output_dir}/train_tasks_record_base.json"
        lib_path = f"{config.file_paths.lib_dir}/library_base.json"
        taxo_path = f"{config.file_paths.lib_dir}/latest_taxonomy_base.json"
        # lib_path = "./data/experience_library/iterations/train_data_4o/library_diag_iter1.json"
        # taxo_path = "./data/experience_library/iterations/train_data_4o/latest_taxonomy_diag_iter1.json"
    else:
        train_data_path = f"{config.file_paths.train_output_dir}/train_tasks_record_diag_iter{start_iter-1}.json"
        lib_path = f"{config.file_paths.lib_dir}/library_refine_iter{start_iter-1}.json"
        taxo_path = f"{config.file_paths.lib_dir}/latest_taxonomy_diag_iter{start_iter-1}.json"
    # Load task recorded previously
    train_tasks = DataLoader(train_data_path, mode="learn", filter_success_num=None, reset=False)
    # Load previous library
    library = ExperienceLibrary.from_json_file(
                    library_path = lib_path,
                    taxonomy_path = taxo_path)
    # library = ExperienceLibrary()
    # Track iteration metrics
    with open(config.file_paths.metrics_log_path, "r") as f:
        metrics_log = json.load(f)


# Run subset
if config.data_slice:
    start = config.data_slice[0]
    end = config.data_slice[1]
    train_tasks = train_tasks.slice(start, end)

start_time = time.time()
for iter in range(start_iter, end_iter): 
    iter_start_time = time.time()
    # Update library retriever
    llm_retri = LibraryRetrieval(lib=library, model=config.base_model, service=config.base_service, temperature=0)

    #* Library online learning for once
    if iter == 0:
        iter_metrics = run_library_online_learning(
            iter, 
            train_tasks, 
            llm_retri, llm_opt_online, llm_diag, llm_ins, library, 
            config.params,
            config.file_paths
        )

        # Save checkpoint
        print(iter_metrics)
        metrics_log.append(iter_metrics)
        save_checkpoint(library=library, tasks=train_tasks, metrics=metrics_log, paths=config.file_paths, suffix="base")
        # directly continue to iter 1
        continue
    #* Library Diagnosis
    iter_metrics = run_library_diagnosis(
        iter, 
        train_tasks, 
        llm_retri, llm_opt, llm_diag, llm_ins, library, 
        config.params,
        config.file_paths,
        max_workers=12
    )
    
    # Save checkpoint
    print(iter_metrics)
    metrics_log.append(iter_metrics)
    save_checkpoint(library=library, tasks=train_tasks, metrics=metrics_log, paths=config.file_paths, suffix=f"diag_iter{iter}")

    #* Library Refinement
    llm_evolve = LibraryEvolution(lib=library, model=config.base_model, service=config.base_service, temperature=0.7)
    refined_library, avg_refinement_rate = run_library_refinement(
        iter=iter, tasks=train_tasks, 
        config=config, llm_evolve=llm_evolve,
        verbose=False, save_data=True, output_path=config.file_paths.train_output_dir,
        max_workers=12
    )
    print("refinement_rate:", avg_refinement_rate)
    # Save iteration metrics log for library evolution phase
    last_metrics = metrics_log[-1]
    last_metrics["refinement_rate"] = round(avg_refinement_rate, 3)

    # #* The chosen best variant for the next round
    library = refined_library
    # Save library
    save_checkpoint(library=library, tasks=None, metrics=metrics_log, paths=config.file_paths, suffix=f"refine_iter{iter}")

    iter_duration = cal_time_cost(iter_start_time, f'Iteration {iter} Total Pipeline')

# Count time cost
total_duration = cal_time_cost(start_time, f'The iterative library learning and evolution process for {config.params.num_iterations} iterations')

# Print structured metrics summary
print_metrics_summary(metrics_log)