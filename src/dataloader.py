import pandas as pd
import json
from typing import Optional 

class Task:
    def __init__(self, task_id, desc, ground_truth, formulation=None, correct_program=None, tag=None, cluster=None):
        self.id = task_id
        self.desc = desc
        self.ground_truth = ground_truth
        self.formulation = formulation
        self.correct_program = correct_program
        self.tag = tag
        self.cluster = cluster

        # Initialize task records
        self.success_count = 0                  # Count of task success
        self.confidence = 0                     # The correct confidence of task
        self.output_status = []                 # List of status for program output per iteration
        self.fail_to_execute = 0               # Count of task failed to correct program
        self.fail_to_verify = 0
        self.retri_ins_lst = []

    def to_dict(self, mode="learn"):
        if mode == "learn":
            # json file content
            return {
                "task_id": self.id,
                "description": self.desc,
                "ground_truth": self.ground_truth,
                "formulation": self.formulation,
                "correct_program": self.correct_program,
                "output_status": self.output_status,
                "success_count": self.success_count,
                "success_confidence": self.confidence,
                "fail_to_execute": self.fail_to_execute,
                "fail_to_verify": self.fail_to_verify,
                "retrieved_insights": self.retri_ins_lst,
                "tag": self.tag,
                "cluster": self.cluster
            }
        elif mode == "test":
            return {
                "task_id": self.id,
                "description": self.desc,
                "ground_truth": self.ground_truth,
                "output_status": self.output_status,
                "tag": self.tag
            }

class DataLoader:
    """
    Loads and holds a list of Task objects
    """
    def __init__(
        self,
        data_path: str = None,
        mode: str = "learn",
        task_list: list = None,
        filter_success_num: int = None,
        reset: bool = False
    ):
        self.tasks = []

        if task_list is not None:
            # Construct from existing Task list
            self.tasks = task_list
        elif data_path:
            # Auto-detect by file extension
            if data_path.endswith('.csv'):
                self._load_from_csv(data_path, mode)
            elif data_path.endswith('.json'):
                # Load JSON with optional filtering and resetting
                self.load_from_json(data_path, filter_success_num=filter_success_num, reset=reset)
            else:
                raise ValueError(f"Unsupported file type for data_path: {data_path}")
    

    def _load_from_csv(self, path: str, mode: str):
        dataset = pd.read_csv(path)
        for _, row in dataset.iterrows():
            task_id = row.get("task_id")
            desc = row.get("description")
            gt = row.get("ground_truth")
            if pd.isna(gt):
                continue
            formulation = row.get("formulation") if mode == "learn" and "formulation" in dataset.columns else None
            code = row.get("optimal_code") if mode == "learn" else None
            self.tasks.append(Task(
                task_id=task_id, 
                desc=desc, 
                ground_truth=gt, 
                formulation=formulation, 
                correct_program=code))
    
    
    def load_from_json(self, path: str, filter_success_num: Optional[int] = None, reset: bool = False):

        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        loaded_tasks = []
        for item in data:
            gt = item.get("ground_truth")
            if (
                gt is None
                or (isinstance(gt, float) and gt != gt)  # NaN
            ):
                continue

            if isinstance(gt, str):
                gt = float(gt)

            task = Task(
                task_id=item.get("task_id"),
                desc=item.get("description"),
                ground_truth=gt,
                correct_program=item.get("correct_program", None),
                tag=item.get("tag"),
                cluster=item.get("cluster")
            )

            # Restore progress
            task.output_status = item.get("output_status", [])
            task.success_count = item.get("success_count", 0)
            task.confidence = item.get("confidence", 0)
            task.fail_to_execute = item.get("fail_to_execute",0)
            task.fail_to_verify = item.get("fail_to_verify", 0)
            task.retri_ins_lst = item.get("retrieved_insights", [])

            # Apply filtering only if a threshold is provided
            if filter_success_num is not None and task.success_count >= filter_success_num:
                continue

            # Optionally reset progress for remaining tasks
            if reset:
                task.output_status    = []
                task.success_count    = 0
                task.confidence       = 0
                task.fail_to_execute = 0
                task.fail_to_verify   = 0
                task.retri_ins_lst = []

            loaded_tasks.append(task)

        self.tasks = loaded_tasks

    def save_as_json(self, path):
        with open(path, "w", encoding="utf-8") as f:
            f.write("[\n")
            total = len(self.tasks)
            for idx, task in enumerate(self.tasks):
                data = task.to_dict()
                # Pop out 'retrieved_insights' and serialize it compactly
                compact = json.dumps(
                    data.pop("retrieved_insights", []), ensure_ascii=False, separators=(",", ":"))

                # Pretty-print the remaining fields with 2-space indent
                pretty_body = json.dumps(data, ensure_ascii=False, indent=2).rstrip("}")

                # Reassemble the object:
                comma = "," if idx < total - 1 else ""
                f.write(
                    f"{pretty_body},\n"
                    f"  \"retrieved_insights\": {compact}\n"
                    f"}}{comma}\n"
                )
            f.write("]\n")


    def save_as_csv(self, path):
        df = pd.DataFrame([task.to_dict() for task in self.tasks])
        df.to_csv(path, index=False)

    def slice(self, start: int, end: int):
        """
        Returns a DataLoader subset object
        """
        return DataLoader(task_list=self.tasks[start:end])
    
    def __getitem__(self, index):
        return self.tasks[index]

    def __len__(self):
        return len(self.tasks)

    def __iter__(self):
        return iter(self.tasks)

    def subset_by_ids(self, id_lst, inplace=False):
        """
        Return a DataLoader containing only tasks whose ids are in the target list
        """
        id_to_task = {task.id: task for task in self.tasks}
        subset = [id_to_task[id] for id in id_lst if id in id_to_task]

        if inplace:
            self.tasks = subset
            return self
        return DataLoader(task_list=subset)