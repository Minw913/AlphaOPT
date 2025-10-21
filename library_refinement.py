import json
import traceback
import copy
from tqdm.auto import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

from src.utils import save_log_data
from src.experience_library import ExperienceLibrary
from src.dataloader import DataLoader 
from src.llm_retriever import LibraryRetrieval
from src.llm_evolver import LibraryEvolution

from src.prompts.prompts_evolve import PROMPT_INS_REFINEMENT


def run_library_refinement(iter, tasks, config, llm_evolve, verbose=False, save_data=False, output_path=None, max_workers=8):
    """
    Parallelize only the outer loop over insights.
    """

    def _process_one_insight(ins):
        """
        Process a single insight:
            1) Collect neg/unr reasons, call LLM to generate conditions per task.
            2) Integrate into a refinement prompt, get K new conditions.
            3) Build K library variants and verify retrieval to choose the best.
        Returns a dict with final condition and distribution to be merged in the main thread.
        """

        # Copy lists to avoid accidental in-place mutation on shared objects
        pos_task_ids = list(ins.distribution.get("positive") or [])
        neg_task_ids = list(ins.distribution.get("negative") or [])
        unr_task_ids = list(ins.distribution.get("unretrieved") or [])

        # If there are neither negative nor unretrieved tasks, skip refinement
        if not neg_task_ids and not unr_task_ids:
            return None

        # Add successful tasks that actually retrieved this insight into positive
        for task in tasks:
            if task.output_status[-1] == "optimal":
                if ins.insight_id in task.retri_ins_lst:
                    pos_task_ids.append(task.id)

        # Generate conditions for negative tasks
        neg_condition_lst = []
        for task in tasks.subset_by_ids(neg_task_ids):
            neg_condition = llm_evolve.generate_neg_condition(task, ins, iter, verbose=verbose, output_dir=output_path)
            neg_condition_lst.append(neg_condition)

        # Generate conditions for unretrieved tasks
        unr_condition_lst = []
        for task in tasks.subset_by_ids(unr_task_ids):
            unr_condition = llm_evolve.generate_unr_condition(task, ins, iter, verbose=verbose, output_dir=output_path)
            unr_condition_lst.append(unr_condition)

        #* Refine insight conditions 
        refined_conditions_k = llm_evolve.refine_insight(iter, neg_condition_lst, unr_condition_lst, ins, config.params.variant_num, verbose=verbose)

        # Build K library variants and evaluate
        library_variants_k = llm_evolve.build_library_variant(ins.insight_id, refined_conditions_k)

        # Baseline success rate before refinement
        total_tasks_num = len(pos_task_ids + neg_task_ids + unr_task_ids)
        original_performance = len(pos_task_ids) / total_tasks_num if total_tasks_num > 0 else 0

        best_performance = original_performance
        best_pos_retri_count = len(pos_task_ids)
        best_neg_retri_count = len(neg_task_ids)
        best_unr_retri_count = len(unr_task_ids)
        best_matched_pos_tids, best_matched_neg_tids, best_matched_unr_tids = [], [], []
        latest_condition = getattr(ins, "condition", None)

        # Evaluate each variant
        for i, lib in enumerate(library_variants_k):
            llm_retri = LibraryRetrieval(lib=lib, model=config.base_model, service=config.base_service, temperature=0)
            pos_retri_count, matched_pos_tids = llm_evolve.verify_retrieval(ins.insight_id, tasks, pos_task_ids, llm_retri)
            neg_retri_count, matched_neg_tids = llm_evolve.verify_retrieval(ins.insight_id, tasks, neg_task_ids, llm_retri)
            unr_retri_count, matched_unr_tids = llm_evolve.verify_retrieval(ins.insight_id, tasks, unr_task_ids, llm_retri)

            # Variant scoring metric: the number of retrieved pos, unr insights and the decrease in neg insights number
            variant_performance = (pos_retri_count + unr_retri_count + len(neg_task_ids) - neg_retri_count) / total_tasks_num if total_tasks_num > 0 else 0

            if variant_performance > best_performance:
                best_performance = variant_performance
                latest_condition = refined_conditions_k[i] if i < len(refined_conditions_k) else latest_condition
                best_pos_retri_count = pos_retri_count
                best_neg_retri_count = neg_retri_count
                best_unr_retri_count = unr_retri_count
                best_matched_pos_tids = matched_pos_tids
                best_matched_neg_tids = matched_neg_tids
                best_matched_unr_tids = matched_unr_tids

        performance_gain = best_performance - original_performance
        # Return a compact result to be merged by the main thread
        return {
            "insight_id": ins.insight_id,
            "orig_condition": getattr(ins, "condition", None),
            "latest_condition": latest_condition,
            "distributions": {
                "positive": best_matched_pos_tids,
                "negative": best_matched_neg_tids,
                "unretrieved": best_matched_unr_tids
                },
            "performance_gain": performance_gain,
            "report": (
                f"\nBest Performance on insight {ins.insight_id}: {best_performance} "
                f"\n Performance Gain: {performance_gain}"
                f"\npositive (before: {len(pos_task_ids)}; after: {best_pos_retri_count}) "
                f"\nnegative (before: {len(neg_task_ids)}; after: {best_neg_retri_count}) "
                f"\nunretrieved (before: {len(unr_task_ids)}; after: {len(unr_task_ids) - best_unr_retri_count})"
            )
        }

    # Results dictionaries (updated only in the main thread; no locks needed)
    refined_insights = {}         # {insight_id: [original_condition, refined_condition]}
    insight_distributions = {}    # {insight_id: {"positive": [...], "negative": [...], "unretrieved": [...]}}

    # Preselect insights to run for proper tqdm progress
    candidate_insights = [ins for ins in llm_evolve.library]
    total = len(candidate_insights)
    refined_ins_num = 0
    total_performance_gain = 0 

    # Thread pool over insights only
    with ThreadPoolExecutor(max_workers=max_workers) as ex, tqdm(total=total, desc=f"[Iteration {iter}] Library Refinement") as pbar:
        future_map = {ex.submit(_process_one_insight, ins): ins.insight_id for ins in candidate_insights}
        for fut in as_completed(future_map):
            pbar.update(1)
            try:
                res = fut.result()
            except Exception:
                traceback.print_exc()
                continue
            # The insight do not have negatvie or unretrieved tasks
            if not res:
                continue

            refined_ins_num += 1 
            total_performance_gain += res["performance_gain"]
            iid = res["insight_id"]
            refined_insights[iid] = [res["orig_condition"], res["latest_condition"]]
            insight_distributions[iid] = res["distributions"]
            # Print once per completed insight to avoid interleaved outputs from threads
            print(res["report"])

    # Optional: persist results
    if save_data and output_path:
        refined_ins_list = [
            {
                "insight_id": insight_id,
                "original_condition": conds[0],
                "refined_condition": conds[1]
            }
            for insight_id, conds in refined_insights.items()
        ]
        save_log_data(refined_ins_list, f"{output_path}/refined_insights_iter{iter}.json")
        save_log_data(insight_distributions, f"{output_path}/refined_insight_distributions_iter{iter}.json")
    
    # Calculate the average refinement again (the average proportion of solved retrieval-misaligned tasks per insight)
    avg_refinement_rate = total_performance_gain / refined_ins_num if refined_ins_num else 0

    # Write refined conditions back to a copied library
    refined_library = copy.deepcopy(llm_evolve.library)
    for ins in refined_library:
        if ins.insight_id in refined_insights:
            ins.condition = refined_insights[ins.insight_id][1]
    return refined_library, avg_refinement_rate
