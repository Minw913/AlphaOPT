import os
import json
from typing import List, Dict, Any, Optional, Callable
from copy import deepcopy
import itertools

class Insight:
    def __init__(self, data: dict):
        self.insight_id = data.get("insight_id")
        self.taxonomy = data.get("taxonomy")
        self.condition = data.get("condition")
        self.explanation = data.get("explanation")
        self.example = data.get("example")
        self.version = data.get("version", 1)
        self.task_id = data.get("task_id")
        self.iteration = data.get("iteration")

        # Counters
        self.occurrence = data.get("occurrence", 0)  # how many times retrieved
        # self.correctness = data.get("correctness", 0)  # how many times led to success

        initial_dist = {"positive": [], "negative": [], "unretrieved": [], "irrelevant": [], "invalid": []}
        self.distribution = data.get("distribution", initial_dist) # how the insight work on target tasks

        # Genealogy (only add if present in the data)
        self.merged_from = data.get("merged_from", None)  # list of parent IDs, default is None
        self.lineage = data.get("lineage", None)  # list of all primitive ancestors, default is None

        # Remove attributes if not provided
        if self.merged_from is None:
            del self.merged_from
        if self.lineage is None:
            del self.lineage

    def to_dict(self) -> dict:
        result = {
            "insight_id": self.insight_id,
            "taxonomy": self.taxonomy,
            "condition": self.condition,
            "explanation": self.explanation,
            "example": self.example,
            "iteration": self.iteration,
            "version": self.version,
            "task_id": self.task_id,
            "occurrence": self.occurrence,    # usage record
            # "correctness": self.correctness,   # usage record
            "distribution": self.distribution
        }

        # Only include these fields if they are present
        if hasattr(self, 'merged_from'):
            result["merged_from"] = self.merged_from
        if hasattr(self, 'lineage'):
            result["lineage"] = self.lineage

        return result
    

class ExperienceLibrary:
    def __init__(self, insight_list: list | None = None):
        """
        Build an ExperienceLibrary from an empty library and a pre-defined taxonomy dictionary
        """
        self._library = []          # type: list[Insight]
        if insight_list:                 # Skip if None or []
            for ins in insight_list:
                self._library.append(Insight(ins))

        with open("./data/experience_library/fewshot_taxonomy.json", "r", encoding="utf-8") as f:
            self.taxonomy = json.load(f)
        # self._taxo_lock = threading.RLock()

    @classmethod
    def from_json_file(cls, library_path: str, taxonomy_path: Optional[str] = None) -> "ExperienceLibrary":
        """
        Read a JSON file (list of dicts) and return an ExperienceLibrary instance.
        """
        if not os.path.isfile(library_path):
            raise FileNotFoundError(library_path)
        with open(library_path, "r") as f:
            data = json.load(f)     # Data is a list[dict]

        inst = cls(insight_list=data)

        if taxonomy_path is not None:
            if not os.path.isfile(taxonomy_path):
                raise FileNotFoundError(taxonomy_path)
            with open(taxonomy_path, "r", encoding="utf-8") as f:
                inst.taxonomy = json.load(f) 

        return inst
        
    def __getitem__(self, index: int) -> Insight:
        return self._library[index]

    def __setitem__(self, index: int, new_insight: Insight):
        self._library[index] = new_insight

    def __len__(self):
        return len(self._library)

    def to_json(self) -> list:
        return [ins.to_dict() for ins in self._library]

    def save(self, path: str):
        """
        Save the current library to a JSON file
        """
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.to_json(), f, indent=2)

    def save_taxonomy(self, path: str):
        """
        Save the current taxonomy to a JSON file
        """
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.taxonomy, f, indent=2, ensure_ascii=False)
        
    def update_usage(self, insight_ids: list, success: bool):
        """
        Increment occurrence for all used insight_ids.
        If `success` is True, also increment correctness.
        """
        # insight_ids = [ins["insight_id"] for ins in update_lst]
        for ins in self._library:
            if ins.insight_id in insight_ids:
                ins.occurrence += 1
                if success:
                    ins.correctness += 1

    def retrieve_insights_by_id(
        self,
        insight_ids: int | list[int],
        *,
        filter_fn: Optional[Callable[["Insight"], bool]] = None,
    ) -> list[dict]:
        """
        Retrieve insights by id(s) and return a list of dicts.
        If filter_fn is provided, only insights passing filter_fn(ins) are kept.
        """
        raw_ids = [insight_ids] if isinstance(insight_ids, int) else list(insight_ids)

        # Flatten nested ID lists
        ids_flat = []
        for iid in raw_ids:
            if isinstance(iid, list):
                ids_flat.extend(iid)
            else:
                ids_flat.append(iid)

        # For quick search, create a dict: id -> Insight
        id2ins = {ins.insight_id: ins for ins in self._library}

        # Gather matching insights in order, and convert to dict
        insights = []
        for iid in ids_flat:
            ins = id2ins.get(iid)
            if ins is None:
                continue

            #  filter_fn : A predicate to filter insights. Example: filter_fn=lambda ins: getattr(ins, "confidence", 0) > 0.7
            if filter_fn is not None and not filter_fn(ins):
                continue

            insights.append({
                "insight_id": ins.insight_id,
                "taxonomy": ins.taxonomy,
                "condition": ins.condition, #TODO maybe not need
                "explanation": ins.explanation,
                "example": ins.example,
            })

        return insights

    def retrieve_by_taxonomy(
        self,
        query_taxonomy: Dict[str, Dict[str, List[str]]],
        filter_fn: Optional[Callable[[Any], bool]] = None,
        include_task_id: bool = False
    ) -> List[Any]:
        """
        Retrieve all insights whose taxonomy contains at least one (stage, level-1, level-2) triple
        that appears in the provided query_taxonomy.
        query_taxonomy : dict
            Shape like:
            {
                "Domain Modeling": {
                    "Resource Allocation": ["Multi-Commodity Flow", ...]
                },
                "General Formulation": {
                    "Variable Definition": ["Continuous vs. Discrete Confusion", ...],
                    "Constraint Formulation": ["Incorrect Relational Operators"]
                },
                "Code Implementation": {
                    "Solver & API Syntax": ["Library Import/Reference Errors"]
                }
                }
        filter_fn : Callable[[Any], bool], optional
        A function that takes an insight and returns True if it should be included.
        If None, no filtering is applied.
        """

        # Build the target set of (stage, level-1, level-2) from query
        wanted = set()
        if isinstance(query_taxonomy, dict):
            for stage, lvl1_map in query_taxonomy.items():
                if not isinstance(lvl1_map, dict):
                    continue
                for lvl1, labels in lvl1_map.items():
                    if isinstance(labels, list):
                        # Query format: {stage: {level1: [level2_list]}}
                        for lbl in labels:
                            wanted.add((stage, lvl1, str(lbl)))
                    elif isinstance(labels, dict):
                        # Query format: {stage: {level1: {level2: value}}}
                        for lbl in labels.keys():
                            wanted.add((stage, lvl1, str(lbl)))
                    # if labels is neither list nor dict, ignore silently

        if not wanted:
            return []  # No valid query triples -> no results

        insights = []

        # Scan the library and test membership
        for ins in self._library:
            # Apply filter if provided
            if filter_fn is not None and not filter_fn(ins):
                continue
            tax = ins.taxonomy or {}
            matched = False

            for stage, lvl1_map in tax.items():
                if not isinstance(lvl1_map, dict):
                    continue
                for lvl1, lvl2_val in lvl1_map.items():
                    # Handle both list format (from query) and dict format (from library)
                    if isinstance(lvl2_val, list):
                        # Query format: {stage: {level1: [level2_list]}}
                        lvl2_list = lvl2_val
                    elif isinstance(lvl2_val, dict):
                        # Library format: {stage: {level1: {level2: {definition, condition}}}}
                        # Handle both dict with values and dict with None values
                        lvl2_list = [k for k, v in lvl2_val.items() if v is not None or k is not None]
                    else:
                        continue

                    # Check if any (stage, lvl1, lvl2) hits
                    for lbl in lvl2_list:
                        if (stage, lvl1, str(lbl)) in wanted:
                            matched = True
                            break
                if matched:
                    break

            if matched:
                # print("match taxonomy!")
                if include_task_id:
                    insights.append({
                        "insight_id": ins.insight_id,
                        "taxonomy": ins.taxonomy,
                        "condition": ins.condition,
                        "task_id": ins.task_id,
                        "iteration": ins.iteration
                    })
                else:
                    # Gather matching insights in order, and convert to dict
                    insights.append({
                        "insight_id": ins.insight_id,
                        "taxonomy": ins.taxonomy,
                        "condition": ins.condition
                    })

        return insights


    def add_insights(self, new_insights: list, iteration:int=None) -> None:
        # Current maximum id in the library (0 if empty)
        max_id = max(
            (ins.insight_id for ins in self._library if ins.insight_id is not None),
            default=0
        )

        for ins in new_insights:
            max_id += 1                      # Assign the next incremental id

            taxo = ins.get("taxonomy", {}) 
            norm = {}
            # lvl1 = stage (e.g., "General Formulation"); lvl1_val = {level1_name: {level2: (null|str)}}
            for lvl1, lvl1_val in taxo.items():
                if not isinstance(lvl1_val, dict):
                    norm[lvl1] = {}
                    continue
                # Map each level1_name to the list of all level2 label names under it.
                norm[lvl1] = {
                    level1_name: (list(level2_dict.keys()) if isinstance(level2_dict, dict) else [])
                    for level1_name, level2_dict in lvl1_val.items()
                }
            ins["taxonomy"] = norm

            # Merge defaults with the incoming dict (incoming keys win if duplicated)
            enriched = {
                **ins,
                "insight_id":  max_id,
                "iteration": iteration,
                "version":     1,
                "occurrence":  0,
                # "correctness": 0
            }

            self._library.append(Insight(enriched))


    @staticmethod
    def _update_one_stage(
        stage: Dict[str, Dict[str, str]],
        ins_label_dict: Dict[str, Any]
    ) -> Dict[str, Dict[str, str]]:
        """
        Merge a single Level-1/Level-2 insight label into one stage of two-level taxonomy
        """

        def _find_ci_key(d: dict, key: str):
            """
            Return the existing key in dictionary that matches in a case-insensitive way
            """
            k_norm = str(key).casefold()
            for k in d.keys():
                if str(k).casefold() == k_norm:
                    return k
            return None
    
        for lvl1, lvl2_spec in (ins_label_dict or {}).items():
            # Find existing Level-1 key ignoring case
            l1_key = _find_ci_key(stage, lvl1)

            # If Level-1 doesn't exist, create it
            if l1_key is None:
                l1_key = lvl1
                stage[l1_key] = {}

            # Only process if Level-2 spec is a dictionary
            if isinstance(lvl2_spec, dict):
                for l2_name, info in lvl2_spec.items():
                    l2_dict = stage[l1_key]

                    # Find existing Level-2 key ignoring case
                    l2_key = _find_ci_key(l2_dict, l2_name)

                    # If not found, add it with given information (definition + condition, empty dictionary if None)
                    if l2_key is None:
                        l2_dict[l2_name] = {} if info is None else info
                    else:
                        # If found, keep the original definition (no overwrite)
                        continue


    def update_taxonomy(self, new_labeled_insights: List[Dict[str, Any]]) -> None:
        """
        Update taxonomy dictionary based on a list of labeled insights
        """
        # Normalize single dict input to list
        if isinstance(new_labeled_insights, dict):
            new_labeled_insights = [new_labeled_insights]

        for ins in new_labeled_insights or []:
            ins_taxo = ins.get("taxonomy", {})
            if not isinstance(ins_taxo, dict):
                continue

            # Domain Modeling
            if "Domain Modeling" in ins_taxo and isinstance(ins_taxo["Domain Modeling"], dict):
                self._update_one_stage(self.taxonomy.setdefault("Domain Modeling", {}),
                                    ins_taxo["Domain Modeling"])

            # General Formulation
            if "General Formulation" in ins_taxo and isinstance(ins_taxo["General Formulation"], dict):
                self._update_one_stage(self.taxonomy.setdefault("General Formulation", {}),
                                    ins_taxo["General Formulation"])

            # Code Implementation
            if "Code Implementation" in ins_taxo and isinstance(ins_taxo["Code Implementation"], dict):
                self._update_one_stage(self.taxonomy.setdefault("Code Implementation", {}),
                                    ins_taxo["Code Implementation"])
    

    def replace_merged_insights(self, existing_insights: List[dict]) -> None:
        """
        Replace existing insights that were merged with new merged insights.
        This removes the old insights from the library.
        
        Args:
            existing_insights: List of existing insights that were merged
        """
        # Get insight IDs that were merged (from existing_insights)
        merged_insight_ids = [ins["insight_id"] for ins in existing_insights]
        
        # Remove the merged insights from library
        self._library = [ins for ins in self._library if ins.insight_id not in merged_insight_ids]
        
        print(f"Removed {len(merged_insight_ids)} existing insights that were merged: {merged_insight_ids}")

    def merge_into_library(
        self,
        all_merged_ids: List[List[int]],
        all_merged_insights: List[dict],
        lib_base: "ExperienceLibrary"
    ) -> "ExperienceLibrary":
        """
        Produce a new library variant by replacing old insights with new merged insights
        """

        # Cache an id → Insight map before deletions
        cache_id2ins = {ins.insight_id: ins for ins in lib_base._library}

        # Remove every insight that will be merged
        ids_to_remove = set(itertools.chain.from_iterable(all_merged_ids))
        lib_base._library = [
            ins for ins in lib_base._library
            if ins.insight_id not in ids_to_remove
        ]

        current_max_id = max((ins.insight_id for ins in lib_base._library), default=0)

        # Append each newly merged insight
        for ids_group, merged_ins in zip(all_merged_ids, all_merged_insights):
            current_max_id += 1

            # Build the full lineage set by unioning every parent's lineage
            lineage_set = set()
            parent_versions = []
            task_ids_set = set()

            # lineage attribute
            for parent_id in ids_group:
                parent_ins = cache_id2ins.get(parent_id)
                # lineage
                if getattr(parent_ins, "lineage", None):
                    lineage_set.update(parent_ins.lineage)
                else:
                    lineage_set.add(parent_id)
                # version
                parent_versions.append(parent_ins.version)
                # source task ids
                if parent_ins.task_id is not None:
                    if isinstance(parent_ins.task_id, list):
                        task_ids_set.update(parent_ins.task_id)
                    else:
                        task_ids_set.add(parent_ins.task_id)

            new_version = max(parent_versions) + 1 # one generation higher than the highest parent version
            source_task_ids  = sorted(task_ids_set) if task_ids_set else None

            merged_insight_enriched = {
                "insight_id":  current_max_id,
                "taxonomy": merged_ins.get("taxonomy", ""),
                "condition":   merged_ins.get("condition", ""),
                "explanation": merged_ins.get("explanation", ""),
                "example":     merged_ins.get("example", ""),
                "distribution": {"positive": [], "negative": [], "unretrieved": [], "irrelevant": [], "invalid": []},
                # default / inherited fields
                "version":     new_version,
                "task_id":     source_task_ids,
                "occurrence":  0,
                # "correctness": 0,
                # genealogy fields
                "merged_from": ids_group,            # direct parents of this merge
                "lineage":     sorted(lineage_set),  # all primitive ancestors
            }

            lib_base._library.append(Insight(merged_insight_enriched))

        return lib_base

# Usage example
if __name__ ==  "__main__":
    # Load the library
    lib_path = "./data/experience_library/iterations/train_data_4o/library_diag_iter1.json"
    taxo_path = "./data/experience_library/iterations/train_data_4o/latest_taxonomy_diag_iter1.json"
    library = ExperienceLibrary.from_json_file(
        library_path = lib_path,
        taxonomy_path = taxo_path)

    print(f"Library loaded with {len(library)} insights")
    
    # Test cases
    test_cases = [
        {
            'General Formulation': {
                'Variable Definition': {
                    'Continuous vs. Discrete Confusion': {
                        'definition': 'Choose integer/binary for indivisible items; continuous for divisible flows.',
                        'condition': 'Applies when decision quantities in the problem represent indivisible counts or choices versus divisible amounts such as flows.'
                    }
                }
            }
        },
        {
            'General Formulation': {
                'Explicit Bounds': None
            }
        },
        {
            'General Formulation': {
                'Unit Inconsistency': {
                    'definition': 'Keep all terms in compatible units to avoid 1000x errors.',
                    'condition': 'Applies when input data come from different unit systems or incompatible measurement scales.'
                }
            }
        },
        {
            'General Formulation': {
                'Variable Definition': {
                    'Continuous vs. Discrete Confusion': None
                }
            }
        },
        {
            'General Formulation': {
                'Constraint Formulation': {
                    'Incorrect Relational Operators': None
                }
            }
        }
    ]
    
    print("\n" + "="*60)
    print("Testing retrieve_by_taxonomy with different formats")
    print("="*60)
    
    for i, test_taxonomy in enumerate(test_cases, 1):
        print(f"\nTest Case {i}:")
        print(f"Query taxonomy: {test_taxonomy}")
        
        # Test retrieve_by_taxonomy
        results = library.retrieve_by_taxonomy(
            query_taxonomy=test_taxonomy, 
            include_task_id=True
        )
        
        print(f"Found {len(results)} matching insights")
        
        if results:
            print("Matching insights:")
            for j, result in enumerate(results[:3]):  # Show first 3 results
                print(f"  {j+1}. Insight ID: {result.get('insight_id', 'N/A')}")
                print(f"     Task ID: {result.get('task_id', 'N/A')}")
                print(f"     Taxonomy: {result.get('taxonomy', {})}")
        else:
            print("  No matching insights found")
        
        print("-" * 40)
    
    print("\nTest completed!")