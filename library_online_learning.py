import os
import time
import json
import copy
import numpy as np
from tqdm.auto import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock
from itertools import combinations

from src.dataloader import DataLoader 
from src.utils import cal_time_cost
from src.train_eval_utils import *


def self_verify_merged_insights(iter, task, llm_opt, new_insights, all_merged_insights, prev_insights):
    """
    Return:
    all_merged_insights: exclude those not verified insights from previous iterated tasks
    """
    # Obtain task-specific insights that is merged or not merged
    merged_ids = {mid for ins in all_merged_insights for mid in ins.get("merged_ids", [])}
    merged_ins = [ins for ins in all_merged_insights if any(n["insight_id"] in ins.get("merged_ids", []) for n in new_insights)]

    #* If not merged insights for this task, skip self-verify
    if not merged_ins:
        return all_merged_insights

    #* Full-subset self-verify: try all possible non-empty combinations of merged_ins
    n = len(merged_ins)
    verified_ins = []
    for k in range(n, 0, -1):  # from largest subset to smallest
        for merged_subset in combinations(merged_ins, k):
            merged_subset = list(merged_subset)
            merged_subset_ids = {mid for ins in merged_subset for mid in ins.get("merged_ids", [])}
            new_ins_not_merge = [ins for ins in new_insights if ins["insight_id"] not in merged_subset_ids]
            # Construct the new insights to verify
            new_insights_to_verify = new_ins_not_merge + merged_subset
            is_verify = self_verify_test(iter, task, llm_opt, new_insights_to_verify, prev_insights)
            if is_verify:
                print(f"The merged insights are verified on {len(merged_subset)} size subset for {task.id}!")
                verified_ins = merged_subset
                break
            else:
                print(f"The merged insights failed to verified on {len(merged_subset)} size subset for {task.id}!")
        if verified_ins:
            break

    # Exclude failed merged insights from all_merged_insights
    failed_ins = [ins for ins in merged_ins if ins not in verified_ins]
    all_merged_insights = [ins for ins in all_merged_insights if ins not in failed_ins]

    return all_merged_insights


def run_library_online_learning(
    iter, 
    train_tasks,
    llm_retri, llm_opt, llm_diag, llm_ins, library, 
    params,
    paths
):
    """
    Run library learning phase on multithread
    """

    lock = Lock()               # Lock to safely update shared variables
    
    # Global counters for online merge rate calculation
    total_online_merge_attempts = 0
    total_online_merge_successes = 0
    def _train_worker(task, taxo_snapshot, train_output_path):
        """
        Parallelize pipeline on the minibatch of tasks (insight retrieval -> formulation generation -> insight retrieval -> program generation -> check optimality -> insight extraction -> insight verification) for training tasks
        Return: (new insights, is_success, is_execution, is_verify)
        """
        success_counts = 0
        prev_insights = []
        status_lst = []
        task_failure_record = [] #  [failed_status, feedback, formulation, program]
        is_success, is_execution, is_verify, is_self_explore = False, True, True, None

        #* Try generating program/solution up to 5 times
        for k in range(1, params.max_solution_attempts + 1):
            #* Retrieve insights (if any) and generate formulation and program
            prev_insights, candidate_formulation, candidate_program, output, runnable, is_time_out = generate_solution_with_retrieval(
                    iter, task, library, llm_retri, llm_opt, 
                    retrieved_insights=prev_insights,
                    output_path=train_output_path, verbose=False, save_data=True
            )
            if k == 1:
                #* Store the first retrieved insights
                fixed_prev_insights = prev_insights
                # Record task
                if prev_insights:
                    retrieved_ins_ids = [ins["insight_id"] for ins in prev_insights if "insight_id" in ins]      
                else:
                    retrieved_ins_ids = []
            else:
                #* Lock the insights retrieved the first time, do not update anymore
                prev_insights = fixed_prev_insights

            # Extreme case: code extraction failed
            if not candidate_program:
                status_lst.append("parse_error")
                continue

            # Check optimality
            is_optimal, output_status, feedback = check_optimality(task=task, output=output, runnable=runnable, is_time_out=is_time_out)

            status_lst.append(output_status)

            if is_optimal:
                success_counts += 1
                if success_counts == params.max_solution_attempts:
                    task.success_count += 1
                    is_success, is_execution, is_verify = True, None, None

            else:
                print(feedback)
                task_failure_record.append([output_status, feedback, candidate_formulation, candidate_program])
        
        #* Record the list of output statuses of 5 times
        task.output_status.append(status_lst)
        task.retri_ins_lst.append(retrieved_ins_ids)
        task.confidence = success_counts / params.max_solution_attempts
        print(f"\n   [Task {task.id}]: Succeeded in {success_counts} of {params.max_solution_attempts} tries.")


        # ============= Generate insights for program and formulation ============== #
        #* Generate insights from failed attempts separately
        task_new_insights = []
        fail_to_execute, fail_to_verify_program_ins, fail_to_verify_formu_ins = 0, 0, 0
        is_need_formu_ins = True
        failures = len(task_failure_record)

        for i in range(0, failures):
            output_status, feedback, candidate_formulation, candidate_program = task_failure_record[i]
            #* First fix program that can not execute and generate program-related insights
            corrected_program, new_feedback, new_program_ins = None, None, []
            if output_status == "run_error":
                is_optimal, runnable, corrected_program, new_feedback = llm_diag.diagnose_program(
                    iter=iter,
                    task=task, 
                    failed_program=candidate_program,
                    feedback=feedback,
                    verbose=False,
                    save_data=True,
                    output_path=train_output_path
                )  
                # If the fix fails, skip insight extraction for this task
                if not runnable:
                    fail_to_execute += 1
                    if fail_to_execute == failures:
                        task.fail_to_execute += 1
                        is_execution, is_verify = False, None
                        break
                    continue

                print(f"\n   [Task {task.id}]: Succeeded to fix a program that failed to execution!")

                for attempt in range(1, params.max_verify_attempts + 1):
                    # Generate insights from code implementation
                    # print(f"\n   [Task {task.id}]: Regenerate new insights for verification at attempt {attempt}!")
                    new_program_ins = llm_ins.generate_insights(
                        iter=iter,
                        task=task,
                        corrected_program=corrected_program, 
                        taxonomy=taxo_snapshot,
                        verbose=False,
                        save_data=True,
                        output_path=train_output_path
                    )
                    if is_optimal:
                        #* Keep only those new insights that can solve its source task when applied back
                        is_verify = self_verify_test(iter, task, llm_opt, new_program_ins, prev_insights, True, train_output_path)

                        if is_verify:
                            # print(f"\n   [Task {task.id}]: The new program insights are self-verified at attempt {attempt}!")
                            task_new_insights.extend(new_program_ins)
                            is_need_formu_ins = False
                            break
                    else:
                        # If not optimal only use new_program_ins for generating runnable programs
                        break
                # The for loop has no break statement (i.e., is_optimal but is_verify never is True)
                else: 
                    fail_to_verify_program_ins += 1  
                    if fail_to_verify_program_ins == failures:
                        is_verify = False
                        task.fail_to_verify += 1
                        is_need_formu_ins = False
                    continue
            
            if is_need_formu_ins:
                #* If the task doesnt have the gold-standard program, enable self-explore
                if not task.correct_program:
                    # only self-explore once # if i == 0:
                    # self-explore from k attempts
                    # Update new feedback and program if debugged
                    candidate_program = corrected_program or candidate_program
                    feedback = new_feedback or feedback
                    is_optimal, gold_standard_program = llm_opt.self_explore(task, candidate_program, feedback)
                    if is_optimal:
                        task.correct_program = gold_standard_program
                        is_self_explore = True
                    else: 
                        is_self_explore = False
                        
                        continue
                        # continue: for next failure of this task try self-explore again
                        # break: only try once self-explore per task using the first failed formulation

                for attempt in range(1, params.max_verify_attempts + 1):
                    #* Extract new insights from the correction
                    new_formu_ins = llm_ins.generate_insights(
                        iter=iter,
                        task=task,
                        failed_formulation=candidate_formulation,
                        taxonomy=taxo_snapshot,
                        verbose=False,
                        save_data=True,
                        output_path=train_output_path
                    )

                    new_insights = new_formu_ins + new_program_ins
                    #* Keep only those new insights that can solve its source task when applied back
                    is_verify = self_verify_test(iter, task, llm_opt, new_insights, prev_insights, True, train_output_path)

                    if is_verify:
                        # print(f"\n   [Task {task.id}]: The new formulation insights are self-verified at attempt {attempt}!")
                        task_new_insights.extend(new_insights)
                        break
                else:
                    fail_to_verify_formu_ins += 1  
                    if fail_to_verify_formu_ins == failures:
                        is_verify = False
                        task.fail_to_verify += 1
                        # print(f"\n   [Task {task.id}]: The new formulation insights failed to self-verify at max attempts!")

        #* Conduct merge on task-specific insights
        for idx, ins in enumerate(task_new_insights):
            ins["insight_id"] = idx

        if len(task_new_insights) > 1:
            print(f"Task {task.id} has {len(task_new_insights)} insights to be merged!")
            task_merged_insights = llm_ins.conduct_insight_merge(candidate_insights=task_new_insights, target=f"Task {task.id}", verbose=False)
        else:
            task_merged_insights = []

        if task_merged_insights:
            #* Obtain self-verified merged insights 
            verified_merged_insights = self_verify_merged_insights(iter, task, llm_opt, task_new_insights, task_merged_insights, prev_insights)
            #* Obtain insights that not merged
            verified_merged_ids = {mid for ins in verified_merged_insights for mid in ins.get("merged_ids", [])}
            task_new_insights_not_merged = [ins for ins in task_new_insights if ins["insight_id"] not in verified_merged_ids]

            for ins in verified_merged_insights: 
                ins.pop("merged_ids", None) 

            task_new_insights = task_new_insights_not_merged + verified_merged_insights

        return task_new_insights, is_success, is_execution, is_verify, is_self_explore

    # Experiment metrics 
    train_success_flags = [False] * len(train_tasks)

    fail_to_execute_lst = []

    iter_verify_count = 0
    iter_verify_success = 0
    fail_to_verify_lst = []

    iter_explore_count = 0
    iter_explore_success = 0
    fail_to_explore_lst = []

    batch_verified_merge_rate = []
    batch_insight_merge_rate = []
    
    train_start_time = time.time()

    for start in range(0, len(train_tasks), params.batch_size):
        batch = train_tasks[start:start + params.batch_size] 
        batch_new_insights = []  # Aggregate the new insights generated in this batch
        batch_train_start_time = time.time()
        with ThreadPoolExecutor(max_workers=params.batch_size) as executor:
            futures = {
                executor.submit(
                    _train_worker,
                    task,
                    copy.deepcopy(library.taxonomy),                # pass a snapshot to avoid concurrent writes
                    os.path.join(paths.train_output_dir, f"task_{task.id}")     # per-task output folder
                ): (start+i, task)
                for i, task in enumerate(batch)
            }

            batch_idx = start // params.batch_size + 1
            for future in tqdm(as_completed(futures), total=len(batch), desc=f"[Iteration {iter}] Library Online Learning Phase Batch {batch_idx} (tasks {start+1}-{start+len(batch)}) \n"):
                # (start+i, task)
                idx, task = futures[future]
                new_insights, is_success, is_execution, is_verify, is_self_explore = future.result()

                if is_success is False and is_verify is not None:
                    iter_verify_count += 1
                    if is_verify:
                        iter_verify_success += 1

                if is_execution is False:
                    fail_to_execute_lst.append(task.id)
                
                if is_verify is False:
                    fail_to_verify_lst.append(task.id)

                if is_self_explore is not None:
                    iter_explore_count += 1
                    if is_self_explore:
                        iter_explore_success += 1
                    else:
                        fail_to_explore_lst.append(task.id)

                if new_insights:
                    # Temporarily store new insights of this batch
                    batch_new_insights.extend(new_insights)

                train_success_flags[idx] = is_success

        #* Once this batch is completed, new insights will be added into the library 
        if batch_new_insights: 
            print(f"Batch {batch_idx} has {len(batch_new_insights)} insights to be merged!")
            with lock: 
                num_all_merged = 0
                if len(batch_new_insights) > 1:
                    #* Conduct insight merge and get merged insight(s) 
                    for idx, ins in enumerate(batch_new_insights):
                        ins["insight_id"] = idx
                    merged_batch_new_insights = llm_ins.conduct_insight_merge(candidate_insights=batch_new_insights, target=f"Batch {batch_idx}", verbose=True) 
                    num_all_merged = len(merged_batch_new_insights)
                else:
                    merged_batch_new_insights = []
                #* Verify the merged insight(s) if any 
                if merged_batch_new_insights: 
                    all_tasks_new_insights = []
                    for task in batch: 
                        # Retrieve previous insights by ids 
                        prev_ins_ids = task.retri_ins_lst[0] if task.retri_ins_lst else []
                        prev_insights = library.retrieve_insights_by_id(prev_ins_ids) if prev_ins_ids else []

                        # Obtain new insights of specific task 
                        # TODO task_new_insights = [ins for ins in batch_new_insights if ins["task_id"] == task.id] 
                        # Handle both single task_id and list of task_ids (for merged insights)
                        task_new_insights = []
                        for ins in batch_new_insights:
                            task_id = ins.get("task_id")
                            if isinstance(task_id, list):
                                if task.id in task_id:
                                    task_new_insights.append(ins)
                            elif task_id == task.id:
                                task_new_insights.append(ins) 
                        
                        #* Self-verify on merged insights
                        # merged_batch_new_insights dynamically exclude those not verified merged insights in this round for the tasks in the next rounds
                        if merged_batch_new_insights:
                            merged_batch_new_insights = self_verify_merged_insights(iter, task, llm_opt, task_new_insights, merged_batch_new_insights, prev_insights)

                    #* Add insights not merged of each task
                    for task in batch:
                        # TODO task_new_insights = [ins for ins in batch_new_insights if ins["task_id"] == task.id]
                        # Handle both single task_id and list of task_ids (for merged insights)
                        task_new_insights = []
                        for ins in batch_new_insights:
                            task_id = ins.get("task_id")
                            if isinstance(task_id, list):
                                if task.id in task_id:
                                    task_new_insights.append(ins)
                            elif task_id == task.id:
                                task_new_insights.append(ins)
                        verified_merged_ids = {mid for ins in merged_batch_new_insights for mid in ins.get("merged_ids", [])}
                        task_new_insights_not_merged = [ins for ins in task_new_insights if ins["insight_id"] not in verified_merged_ids]
                        all_tasks_new_insights.extend(task_new_insights_not_merged)
                    
                    for ins in merged_batch_new_insights: 
                        ins.pop("merged_ids", None) 
                    #* Add merged_insights that self-verified on all tasks
                    all_tasks_new_insights.extend(merged_batch_new_insights)
                else:
                    all_tasks_new_insights = batch_new_insights

                print(f"Batch {batch_idx} has {len(all_tasks_new_insights)} insights after merge!")

                #* Online merge with existing library insights for each new insight
                online_merge_success_count = 0
                
                # Update global counters
                total_online_merge_attempts += len(all_tasks_new_insights)
                
                for new_insight in all_tasks_new_insights:
                    # Conduct online merge with existing library insights
                    # import pdb; pdb.set_trace()

                    merged_insights, _, existing_insights = llm_ins.conduct_insight_online_merge(
                        new_insight=[new_insight],  # Wrap in list as expected by the method
                        library=library,
                        verbose=True
                    )
                    
                    # If no merge occurred, add original insight directly
                    if not merged_insights:
                        library.add_insights([new_insight], iter)
                        library.update_taxonomy(new_insight)
                        print(f"No merge occurred for task {new_insight['task_id']}, adding original insight!")
                        continue
                    
                    # If merge occurred, verify merged insights
                    merged_task_ids = merged_insights["task_id"]
                    target_tasks = train_tasks.subset_by_ids(merged_task_ids)
                    all_tasks_verified = True
                    
                    # Verify merged insights on each related task
                    for task in target_tasks:
                        # Get prev_insights for this specific task
                        prev_ins_ids = task.retri_ins_lst[0] if task.retri_ins_lst else []
                        prev_insights = library.retrieve_insights_by_id(prev_ins_ids) if prev_ins_ids else []
                        
                        # Check if this task is the new_insight's task_id
                        new_insight_task_id = new_insight.get('task_id')
                        is_new_insight_task = False
                        if isinstance(new_insight_task_id, list):
                            is_new_insight_task = task.id in new_insight_task_id
                        else:
                            is_new_insight_task = task.id == new_insight_task_id
                            
                        if is_new_insight_task:
                            # For new_insight's task_id: use all new insights for this task (excluding current) + merged_insights
                            # Handle both single task_id and list of task_ids (for merged insights)
                            task_new_insights = []
                            for ins in all_tasks_new_insights:
                                if ins != new_insight:
                                    task_id = ins.get('task_id')
                                    if isinstance(task_id, list):
                                        if task.id in task_id:
                                            task_new_insights.append(ins)
                                    elif task_id == task.id:
                                        task_new_insights.append(ins)
                            task_new_insights.append(merged_insights)
                            
                        else:
                            # For existing insights' task_id: use library insights from iteration 0, excluding merged ones
                            task_new_insights = []
                            for ins in library:
                                if ins.iteration == 0:
                                    # Handle both single task_id and list of task_ids (for merged insights)
                                    task_id = ins.task_id
                                    if isinstance(task_id, list):
                                        if task.id in task_id:
                                            # Check if this insight was merged (excluded from merged_insights)
                                            if ins not in existing_insights:
                                                task_new_insights.append(ins.to_dict())
                                    elif task_id == task.id:
                                        # Check if this insight was merged (excluded from merged_insights)
                                        if ins not in existing_insights:
                                            task_new_insights.append(ins.to_dict())
                            task_new_insights.append(merged_insights)

                        # print("task_new_insights", task_new_insights)
                        is_verify = self_verify_test(iter=None, task=task, llm_opt=llm_opt, new_insights=task_new_insights, prev_insights=prev_insights)
                        if not is_verify:
                            all_tasks_verified = False
                            break
                    
                    # Only add merged insights if all tasks are verified successfully
                    if all_tasks_verified:
                        # Replace existing insights that were merged and add merged insights immediately
                        library.replace_merged_insights(existing_insights)
                        library.add_insights([merged_insights], iter)
                        library.update_taxonomy(merged_insights)
                        online_merge_success_count += 1
                        total_online_merge_successes += 1
                        print(f"Successfully merged insight for task {new_insight['task_id']} with existing library insights!")
                    else:
                        # If verification fails, add original insight immediately
                        library.add_insights([new_insight], iter)
                        library.update_taxonomy(new_insight)
                        print(f"Online merge failed for task {new_insight['task_id']}, adding original insight!")

                # Calculate online merge rate for this batch
                batch_online_merge_rate = (online_merge_success_count / len(all_tasks_new_insights)) if all_tasks_new_insights else 0
                print(f"Batch {batch_idx} online merge rate: {batch_online_merge_rate:.3f} ({online_merge_success_count}/{len(all_tasks_new_insights)})")

                # Calculate verified merge rate: number of successfully verified merged insights / total proposed merged insights
                verified_merge_rate = (len(merged_batch_new_insights) / num_all_merged) if num_all_merged > 0 else None
                batch_verified_merge_rate.append(verified_merge_rate)

                # Calculate insight merge rate: number of merged insights / total original insights
                ins_merge_rate = (len(batch_new_insights) - len(all_tasks_new_insights)) / len(batch_new_insights) if batch_new_insights else None
                batch_insight_merge_rate.append(ins_merge_rate)

                # Periodically save library snapshot to prevent data loss
                library.save(f"{paths.lib_dir}/library_base_snap.json")
                library.save_taxonomy(f"{paths.lib_dir}/latest_taxonomy_base_snap.json")
                train_tasks.save_as_json(f"{paths.train_output_dir}/train_tasks_record_base_snap.json")

        batch_train_duration = cal_time_cost(batch_train_start_time, f'Iteration {iter} Library Online Learning Phase Batch {batch_idx} [{start+1}-{start+len(batch)}]')

    train_duration = cal_time_cost(train_start_time, f'Iteration {iter} Library Online Learning Phase')

    # Calculate the success rate for this iteration
    number_of_train_failures = len(train_success_flags) - sum(train_success_flags)
    train_accuracy = sum(train_success_flags) / len(train_success_flags) if train_success_flags else 0
    verify_success_rate = (iter_verify_success / iter_verify_count) if iter_verify_count > 0 else 0
    explore_success_rate = (iter_explore_success / iter_explore_count) if iter_explore_count > 0 else 0
    
    # Calculate batch-level metrics using np.nanmean for consistent handling of None values
    verified_merge_rate = np.nanmean([x if x is not None else np.nan for x in batch_verified_merge_rate])
    insight_merge_rate = np.nanmean([x if x is not None else np.nan for x in batch_insight_merge_rate])
    
    # Calculate overall online merge rate
    overall_online_merge_rate = (total_online_merge_successes / total_online_merge_attempts) if total_online_merge_attempts > 0 else 0
    
    # Record library learning success log
    iter_metrics = {
        "stage": "Library Online Learning",
        "iter": iter,
        "train_accuracy": round(train_accuracy, 3),
        "library_size": len(library),
        "number_of_train_failures": number_of_train_failures,
        "self_verify_rate": round(verify_success_rate, 3),
        "self_explore_success_rate": round(explore_success_rate, 3),
        "insight_merge_rate": round(insight_merge_rate, 3),
        "verified_merge_rate": round(verified_merge_rate, 3),
        "online_merge_rate": round(overall_online_merge_rate, 3),
        "number_of_train_tasks": len(train_tasks),
        "fail_to_execute_task_ids": fail_to_execute_lst,
        "fail_to_explore_task_ids": fail_to_explore_lst,
        "fail_to_verify_task_ids": fail_to_verify_lst,
        "online_learning_duration (min)": train_duration,
    }

    return iter_metrics