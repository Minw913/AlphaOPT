import os
import time
import json
import copy
from tqdm.auto import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock

from src.dataloader import DataLoader 
from src.utils import cal_time_cost
from src.train_eval_utils import *

def run_library_diagnosis(
    iter, 
    train_tasks,
    llm_retri, llm_opt, llm_diag, llm_ins, library,
    params, 
    paths, 
    max_workers = 8,
):
    """
    Run library insight diagnosis through doing tasks
    """

    lock = Lock()               # Lock to safely update shared variables
    

    def _train_worker(task, taxo_snapshot, library_snapshot, train_output_path):
        """
        Parallelize the entire per-task pipeline (insight retrieval -> formulation generation -> insight retrieval -> program generation -> diagnosis -> insight extraction -> insight verification) for training tasks
        Return: (new insights, is_success, is_execution, is_verify, is_diagnosis)
        """

        #* Retrieve insights (if any) and generate formulation and program
        prev_insights, candidate_formulation, candidate_program, output, runnable, is_time_out = generate_solution_with_retrieval(
                iter, task, library_snapshot, llm_retri, llm_opt, 
                retrieved_insights=[],
                output_path=train_output_path, verbose=False, save_data=True
        )
        # Extreme case: code extraction failed
        if not candidate_program:
            task.output_status.append("parse_error")
            return [], False, False, None, None # new insights, is_success, is_execution, is_verify, is_diagnosis

        # Check optimality
        is_optimal, output_status, feedback = check_optimality(task=task, output=output, runnable=runnable, is_time_out=is_time_out)

        # Record task
        retrieved_ins_ids = [ins["insight_id"] for ins in prev_insights if "insight_id" in ins]             
        task.retri_ins_lst.append(retrieved_ins_ids)
        task.output_status.append(output_status)

        if is_optimal:
            print(f"\n   [Task {task.id}]: Output was optimal. Task succeeds!")
            return [], True, None, None, None  # new insights, is_success, is_execution, is_verify, is_diagnosis
        else:  
            print(feedback)

        # ============= Library Diagnosis ==============
        #* Diagnose the failure and retrieved insights (the cause of failure)
        is_need_formulation_diag = (output_status != "run_error")
        new_program_ins = []

        # First fix program that can not execute
        if output_status == "run_error":
            is_optimal, runnable, corrected_program, _ = llm_diag.diagnose_program(
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
                task.fail_to_execute += 1
                return [], False, False, None, None  # new insights, is_success, is_execution, is_verify, is_diagnosis

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
                
                # if isinstance(new_program_ins, dict):
                # print("generate_insights new_program_ins:", new_program_ins)

                if is_optimal:
                    #* Keep only those new insights that can solve its source task when applied back
                    is_verify = self_verify_test(iter, task, llm_opt, new_program_ins, prev_insights, False, train_output_path)
                    if is_verify:
                        # print(f"\n   [Task {task.id}]: The new insights are self-verified at attempt {attempt}!")
                        return new_program_ins, False, True, True, None  # new insights, is_success, is_execution, is_verify, is_diagnosis
                else:
                    # If not optimal, no need to verify
                    is_need_formulation_diag = True
                    break  

            else:
                # The for loop has no break statement (i.e., is_verify never is True)
                return [], False, True, False, None


        if is_need_formulation_diag:
            # Convert insight
            formulation_ins, program_ins = divide_insight(prev_insights)
            # if isinstance(program_ins, dict):
            # print(" if is_need_formulation_diag: program_ins:", program_ins)
            prev_insights = [formulation_ins, program_ins]

            if new_program_ins:
                # if isinstance(new_program_ins, dict):
                # print("if new_program_ins: new_program_ins:", new_program_ins)
                #* If new program insights are generated, update them into the previous insights
                prev_insights[1] = new_program_ins

            #* Only when retrieved insights for formulation exist
            new_formulation = candidate_formulation
            updated_formulation_ins = []
            if  prev_insights[0]:
                insights_diag, updated_formulation_ins, is_generate_new, new_formulation = llm_diag.diagnose_formulation(
                    iter=iter,
                    task=task, 
                    feedback=feedback,
                    failed_formulation=candidate_formulation,
                    retrieved_insights=prev_insights,
                    llm_opt=llm_opt,
                    llm_retri=llm_retri,
                    verbose=True,
                    save_data=True,
                    output_path=train_output_path
                )  

                if insights_diag:
                    #* For each retrieved insight, append its state with this task
                    for ins in library:
                        if ins.insight_id in insights_diag.keys():
                            # Deduplicate {insight_id: unretrieved}
                            state = insights_diag[ins.insight_id]
                            if task.id not in ins.distribution[state]:
                                ins.distribution[state].append(task.id)

            else: 
                is_generate_new = True

            if is_generate_new:
                for _ in range(1, params.max_verify_attempts + 1):
                    #* Extract new insights from the correction
                    new_insights = llm_ins.generate_insights(
                        iter=iter,
                        task=task,
                        failed_formulation=new_formulation,
                        taxonomy=taxo_snapshot,
                        verbose=False,
                        save_data=True,
                        output_path=train_output_path
                    )

                    if new_program_ins: 
                        new_insights = new_insights + new_program_ins

                    #* Update the formulation insights with positive insights and unretrieved insights
                    prev_insights = [updated_formulation_ins, program_ins]
                    
                    prev_insights = prev_insights[0] + prev_insights[1]
                    #* Keep only those new insights that can solve its source task when applied back
                    is_verify = self_verify_test(iter, task, llm_opt, new_insights, prev_insights, False, train_output_path)

                    if is_verify:
                        print(f"The new generated insights of {task.id} are successfully verified!")
                        # import pdb; pdb.set_trace()
                        # print("new_insights", new_insights)
                        return new_insights, False, True, True, False # new insights, is_success, is_execution, is_verify, is_diagnosis

                # Reach the maximum retries
                return [], False, True, False, False

            else:
                # After insight diagnosis, No new insights are needed
                return [], False, True, None, True


    # Experiment metrics 
    # temp_lib = []
    train_success_flags = [False] * len(train_tasks)

    iter_verify_count = 0
    iter_verify_success = 0

    iter_diagnose_count = 0
    iter_diagnose_success = 0

    fail_to_verify_lst = []
    fail_to_execute_lst = []


    # Initialize new_insights queue and lock for serial processing to avoid version conflicts
    from queue import Queue
    from queue import Empty
    import threading
    import time
    new_ins_queue = Queue()
    queue_lock = Lock()
    processing_active = True
    processed_count = 0
    
    # Counters for online merge rate calculation
    total_new_insights = 0
    successful_merges = 0


    def process_insights_queue():
        nonlocal processed_count, total_new_insights, successful_merges, processing_active
        # print("Start processing insights queue")
        while processing_active or not new_ins_queue.empty():
            try:
                # wait briefly for work; don't die if it's empty yet
                # print("Learning new_ins_queue", new_ins_queue.qsize())
                new_insights = new_ins_queue.get(timeout=0.2)
            
            except Empty:
                # no work yet; loop again while producer is still active
                time.sleep(0.1)
                continue

            if not new_insights:
                new_ins_queue.task_done()
                continue

            for new_insight in new_insights:
                total_new_insights += 1

                # print(f"   [DEBUG] Processing new insight with task_id: {new_insight.get('task_id', 'unknown')}")
                # if library:
                #     print("yes")
                merged_insights, merged_task_to_iter, existing_insights = llm_ins.conduct_insight_online_merge(
                    new_insight=[new_insight],
                    library=library,
                    verbose=True
                )
                # if len(merged_insights) > 0:
                # print(f"   [DEBUG] Merge result: {1 if merged_insights else 0} merged insights, {len(existing_insights)} existing insights")

                # PDB breakpoint - handles multi-threading by stopping all threads
                # import pdb; pdb.set_trace()

                # If no merge occurred, add original insight directly
                if not merged_insights:
                    with lock:
                        library.add_insights([new_insight], iter)
                        library.update_taxonomy(new_insight)
                    print(f"No merge occurred for task {new_insight['task_id']} , adding original insight!")
                    continue

                # If merge occurred, verify merged insights
                merged_task_ids = merged_insights["task_id"] if merged_insights else []
                target_tasks = train_tasks.subset_by_ids(merged_task_ids)

                all_tasks_verified = True
                for task in target_tasks:
                    # Initialize prev_ins_ids for each task separately
                    prev_ins_ids = []
                    task_iter = merged_task_to_iter.get(task.id)
                    if task_iter == -1 and task.retri_ins_lst:
                        prev_ins_ids.extend(task.retri_ins_lst[-1])
                    elif task.retri_ins_lst and len(task.retri_ins_lst) > task_iter:
                        prev_ins_ids.extend(task.retri_ins_lst[task_iter])
                    elif task.retri_ins_lst:
                        prev_ins_ids.extend(task.retri_ins_lst[-1])
                        
                    prev_insights = library.retrieve_insights_by_id(prev_ins_ids) if prev_ins_ids else []
                    # print("prev_insights", prev_insights)
                    # print("   [DEBUG] is_verify: start to verify")
                    # Get insights for verification based on task_iter
                    if task_iter == -1:
                        # For new_insight's task_id: use all new insights for this task (excluding current) + merged_insights
                        #TODO task_new_insights = [ins for ins in new_insights if ins.get('task_id') == task.id and ins != new_insight]
                        # Handle both single task_id and list of task_ids (for merged insights)
                        task_new_insights = []
                        for ins in new_insights:
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
                            # if ins.task_id == task.id and ins.iteration == task_iter:
                            #     # Check if this insight was merged (excluded from merged_insights)
                            #     if ins not in existing_insights:
                            #         task_new_insights.append(ins.to_dict())
                            # TODO
                            if ins.iteration == task_iter:
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
                    
                    is_verify = self_verify_test(iter=None, task=task, llm_opt=llm_opt,
                                                new_insights=task_new_insights, prev_insights=prev_insights)

                    # print(f"   [DEBUG] is_verify: {is_verify}")
                    if not is_verify:
                        all_tasks_verified = False
                        break

                if all_tasks_verified:
                    # print(f"   [DEBUG] SUCCESS: Merging 1 insight after verification")
                    with lock:
                        library.replace_merged_insights(existing_insights)
                        library.add_insights([merged_insights], iter)
                        library.update_taxonomy(merged_insights)
                    successful_merges += 1
                    print(f"Successfully merged insight for task {new_insight['task_id']} with existing library insights!")
                else:
                    # print(f"   [DEBUG] NO MERGE: Adding individual insight. all_tasks_verified={all_tasks_verified}, merged_insights_count={1 if merged_insights else 0}")
                    with lock:
                        library.add_insights([new_insight], iter)
                        library.update_taxonomy(new_insight)
                    print(f"Online merge failed for task {new_insight['task_id']}, adding original insight!")

                processed_count += 1
                if processed_count % 10 == 0:
                    try:
                        library_checkpoint = copy.deepcopy(library)
                        library_checkpoint.save(f"{paths.lib_dir}/library_iter{iter}_diag_snap.json")
                        library_checkpoint.save_taxonomy(f"{paths.lib_dir}/latest_taxonomy_iter{iter}_snap.json")
                        train_tasks.save_as_json(f"{paths.train_output_dir}/train_tasks_record_iter{iter}_snap.json")
                        print(f"[Iteration {iter}] Saved library snapshot after processing {processed_count} insights")
                    except Exception as e:
                        print(f"[Iteration {iter}] Warning: Failed to save snapshot: {e}")

            new_ins_queue.task_done()

    # Start the async processing thread
    processing_thread = threading.Thread(target=process_insights_queue, daemon=True)
    processing_thread.start()

    # Training phase
    train_start_time = time.time()
    # Create a snapshot of the library to avoid data race conditions in parallel threads
    library_snapshot = copy.deepcopy(library)
    with ThreadPoolExecutor(max_workers=4) as executor:   #max_workers
        futures = {
            executor.submit(
                _train_worker,
                task,
                copy.deepcopy(library.taxonomy),  # pass a snapshot to avoid concurrent writes
                library_snapshot,  # pass library snapshot to avoid data race
                os.path.join(paths.train_output_dir, f"task_{task.id}"),     # per-task output folder
            ): (idx, task)
            for idx, task in enumerate(train_tasks)
        }

        for future in tqdm(as_completed(futures), total=len(train_tasks), desc=f"[Iteration {iter}] Library Diagnosis Phase\n"):
            idx, task = futures[future]
            new_insights, is_success, is_execution, is_verify, is_diagnosis = future.result()

            if is_success is False and is_verify is not None:
                iter_verify_count += 1
                if is_verify:
                    iter_verify_success += 1

            if is_execution is False:
                fail_to_execute_lst.append(task.id)

            if is_verify is False:
                fail_to_verify_lst.append(task.id)

            if is_diagnosis is not None:
                iter_diagnose_count += 1
                if is_diagnosis is True:
                    iter_diagnose_success += 1

            train_success_flags[idx] = is_success
            if new_insights:
                # Add new_insights to queue to avoid version conflicts during parallel merging
                # print("new_insights", new_insights)
                with queue_lock:
                    print('Start adding new insights')
                    new_ins_queue.put(new_insights)
                    print(f"Added new insights to queue! Now the queue has length {new_ins_queue.qsize()}")

    
    train_duration = cal_time_cost(train_start_time, f'Iteration {iter} Library Diagnosis Phase')

    # Stop the async processing and wait for queue to be empty
    print(f"[Iteration {iter}] Stopping async processing and waiting for queue to be empty...")
    processing_active = False
    
    # Wait for all items in queue to be processed
    print(f"[Iteration {iter}] Waiting for queue to be empty...")
    try:
        # Use join() with a simple timeout wrapper
        import signal
        def timeout_handler(signum, frame):
            raise TimeoutError("Queue join timeout")
        
        signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(1800)  # 30 minute timeout
        new_ins_queue.join()
        signal.alarm(0)  # Cancel the alarm
        print(f"[Iteration {iter}] Queue is now empty")
    except TimeoutError:
        print(f"[Iteration {iter}] WARNING: Queue join timeout after 30 minutes, forcing continue")
    except Exception as e:
        print(f"[Iteration {iter}] Error waiting for queue: {e}")
    
    # Wait for processing thread to finish with 5 minute timeout
    processing_thread.join(timeout=300)  # 5 minute timeout
    
    print(f"[Iteration {iter}] Queue processing completed, processed {processed_count} insights")
    
    # Update llm_retri with the latest library state
    llm_retri.library = library

    # Calculate the success rate for this iteration
    number_of_train_failures = len(train_success_flags) - sum(train_success_flags)
    train_accuracy = sum(train_success_flags) / len(train_success_flags) if train_success_flags else 0

    # Calculate the self verification success rate of new generated insights
    verify_success_rate = (iter_verify_success / iter_verify_count) if iter_verify_count > 0 else 0

    # Calculate task success rate after diagnosing and resolve problematic insights (no need to generate new insights)
    diagnose_success_rate = (iter_diagnose_success / iter_diagnose_count) if iter_diagnose_count > 0 else 0

    # Calculate online merge success rate: successfully merged insights / total new insights
    online_merge_rate = (successful_merges / total_new_insights) if total_new_insights > 0 else 0

    # Record library learning success log
    iter_metrics = {
        "stage": "Library Diagnosis",
        "iter": iter,
        "train_accuracy": round(train_accuracy, 3),
        "library_size": len(library),
        "number_of_train_failures": number_of_train_failures,
        "self_verify_rate": round(verify_success_rate, 3),
        "diagnose_success_rate": round(diagnose_success_rate, 3),
        "online_merge_rate": round(online_merge_rate, 3),
        "number_of_train_tasks": len(train_tasks) if train_tasks else 0,
        "fail_to_verify_task_ids": fail_to_verify_lst,
        "fail_to_execute_task_ids": fail_to_execute_lst,
        "library_diagnosis_duration (min)": train_duration
    }

    return iter_metrics