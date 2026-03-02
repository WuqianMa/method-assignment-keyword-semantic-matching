"I have two JSON files: `l1_methods.json` (categories) and `l2_methods.json` (sub-categories). The L2 file links to L1 via the `level_1_label` field. Both files contain `semantic_meaning` and `keywords`. Refer to this structure for all data processing tasks to avoid full-file inspection."

---

# Data Schema Reference

## 1. `l1_methods.json`

**Purpose:** High-level methodological branches with broad definitions.

* **Structure:** `Array<Object>`
* **Fields:**
* `name` (String): The category title (e.g., "Empirical and Econometric Methods").
* `semantic_meaning` (String): A formal definition of the branch's application.
* `keywords` (Array<String>): Representative techniques within this branch.



## 2. `l2_methods.json`

**Purpose:** Granular sub-methods mapped back to L1 categories.

* **Structure:** `Array<Object>`
* **Fields:**
* `level_1_label` (String): Foreign key matching `l1_methods.json` -> `name`.
* `method_name` (String): Specific sub-category (e.g., "Difference-in-Differences (DiD) & Event Studies").
* `keywords` (Array<String>): Specific technical terms and model types.
* `semantic_meaning` (String): Detailed application-specific description.



