

description: Strict schema and logic for extracting Knowledge Graph nodes (Domain, Concept, Stakeholder) and relations from EU AI Act and KR AI Law.
globs: ["**/*.json"]
alwaysApply: false
---

# Senior Legal KG Engineer Extraction Schema (Ling Long Protocol)

## 1. Node Definitions & Properties

### (Domain)
* **Properties:** `name`, `description`, `source_file`, `updated_at`
* **Allowed Scopes:** Biometrics, Critical Infrastructure (KR-specific: Nuclear/Water), Education, Employment, Essential Private/Public Services, Healthcare, Transportation, Law Enforcement, Migration, Administration of Justice.

### (Concept)
* **Properties:** `name`, `description`, `lang` (en/ko), `source_file`, `updated_at`
* **Constraint:** ONLY Title Form (e.g., `AI System`). DO NOT append source tags like "(AIAct)" or "(AICoreLaw)" to the Concept name. The name must be clean. No source tags in name. No daily nouns. ONLY extract legally defined terms as Concepts. Do NOT extract specific use-case examples, crimes (e.g., murder, kidnapping), or general daily nouns. MUST specify the 'lang' property ('en' or 'ko') for every Concept.


* **Key Terms:** AI System, High-Risk AI System, Provider, Deployer, GPAI Model, AI Developer, AI Business Operator, Generative AI, Intended Purpose, Substantial Modification, User, Impacted Person, Domestic Agent.

### (Stakeholder)
* **Properties:** `name`, `description`, `source_file`, `updated_at`
* **Constraint:** Use Title Form. Singular only (e.g., `Provider`). If the name matches a `Concept.name`, use the EXACT same string. 
* **Regularization:** `Stakeholder.name` should not distinguish singular or plural, small or capital letters

### (Regulation)
* **Reguirement:** `jurisdiction`, `name`, `effective_date`
* **Explanation:** `Regulation.jurisdiction` indiciates a nation or an organizational body abbreviations such as "KR" or "EU". 

### (Article) 
* **Requirement:** `id`, `full_text`, `metadata`, `title`, `source_file`, `updated_at`
* **Constraint:** `Article.id` should include jurisdiction info + the name of a law + Article number. For example “Korea AI Law::Article 6”. 
* **Note:** `Article.title` is optional. 

### (Rationale)
* **Requirement:** `id`, `full_text`, `source_file`, `title`, `updated_at`
* **Note:** `Rationale.title` is optional. 

### (Annex)
* **Requirement:** `id`, `full_text`, `source_file`, `title`, `updated_at`
* **Note:** `Annex.title` is optional. 

### (Sanction)
* **Requirement:** `name`, `description`, `amount`, `type`, `source_file`, `updated_at`
* **Explanation:** `Sanction.name` is a keyword of the sanction. 
`Sanction.amount` denotes the length of the application of one sanction or the amount of fines. `Sanction.type` denotes the type of a sanction, classifying itself as Financial, Operational, Reputational. `Sanction.description` includes the relevant part of text. 

### (Requirement)
* **Requirement:** `name`, `description`, `source_file`, `updated_at`
* **Explanation:** `Requirement.name` is a keyword of the sanction.  `Requirement.description` includes the full text of the relevant article. 

### (TechCriterion) 
* **Requirement:** `name`, `detail`, `threshold_value`, `unit`, `source_file`, `updated_at`
* **Explanation:** `TechCriterion` specifically indicates a criterion that decides whether to apply a sanction or a requirement. It can be a cumulative amount of computation(FLOPs) during training or inference, or the number of sales/user counts. `TechCriterion.threshold_value` indicates the exact figure of the criterion. `TechCriterion.unit` indicates the unit of the threshold. *TechCriterion.detail` s the relevant part of the text. 

### (UsageCriterion)
* **Requirement:** `name`, `detail`, `threshold_value`, `unit`, `source_file`, `updated_at`
* **Explanatation:** `UsageCriterion.detail` is the the relevant part ofthe text. 


### (RiskCategory)
* **Requirement:** `level`, `description`
* **Consrtaint:** `RiskCategory.level` has only one possible attribute. “High-Risk AI”. It should also be identical to one of Concepts’ Concept.name, which is “High-Risk AI” or "High-Impact AI". 

## 2. Relationship Extraction Logic 

| Relationship | Logic / Trigger Keywords | Start Node | End Node | 
| :--- | :--- | :--- | :--- | 
| `[:MANDATED_FOR]` | Phrases: "shall", "must", "required to", "obliged to". | Start Node: `Requirement` | End Node: `Stakeholder` | 
| `[:APPLIES_TO]` | A concerning people or body who are responsible for a certain Sanction(NEW). | Start Node: `Sanction` | End Node: `Stakeholder` | 
| `[:PERMITS]` | Phrases: "may", "allow", "be entitled to". | Start Node: `Requirement` | End Node: `Stakeholder` | 
| `[:DEFINES]` | Primary legal definition in an Article. | Start Node: `Article` | End Node: `Concept` |
| `[:DETAILS_DEFINITION_OF]` | Detailed explanation on a legal term of a specific Concept. | Start Node: `Rationale` or `Annex` | End Node: `Concept` |
| `[:DETAILS_SCOPE]` | Geographical/material boundaries ("applies to", "within scope"). | Start Node: `Rationale` or `Annex` | End Node: `Domain` |
| `[:JUSTIFIES_LOGIC]` | Legislative purpose ("aims to", "objective", "in order to"). | Start Node: `Rationale` or `Annex` | End Node: `Article`
| `[:INCLUDES]` | Illustrates hierarchy | Start Node: `Regulation` | End Noe: `Article` or `Rationale` or `Annex` |
| `[:ESTABLISHES]` | Explanation on the way to decide if a regulation applies or not | Start Node: `Article` or `Rationale` or `Annex` | End Node: `TechCriterion` or `UsageCriterion`
| `[:TRIGGERS]` | Instance where certain domains trigger a legal action | Start Node: `Domain` | End node: `Requirement` or `Sanction` |
| `[:LEADS_TO]` | Instance where a domain or a criterion indicates High-Risk of High-Impact AI | Start Node: `Domain` or `TechCriterion` or `UsageCriterion` | End Node: `RiskCategory` |
| `[:IMPOSES]` | Instance where a text states a legal obligations to a concerning people. | Start Node: `Article` | End Node: `Requirement` | 
| `[:PENALIZES_WITH]` | Instance where a text states a legal sanctions to a concerning people. | Start Node: `Article` | End Node: `Sanction` |
| `[:IS_A]` | Instance where a text states a certain support to boost AI regarding activities. | Start Node: `Article` | End Node: `Support` | 
| `[:ENCOMPASSES]`| Concept from one jurisdiction is broader and logically includes a specific concept from another jurisdiction | Start node: `(c1:Concept)` | End Node `(c2:Concept)` | 
| `[:SUPPLEMENTS]` | One legal document from one jurisdiction detailed criteria or lists that fill the regulatory gaps of a broader concept from another jurisdiction. | Start node: `(Article/Annex/Rationale)` | End Node: `Concept` |


## 3. General Constraints (VERY Critical)
1.  **Verbatim Property:** EVERY relationship MUST include a `description` property with the original text excerpt. Specifically, make sure `(Sanction) is connected to `(Stakeoholder) via `[:APPLIES_TO]` as `(Sanction)-[:APPLIES_TO]-(Stakeholder)`.
2.  **Conflict Resolution:** If "apply to" is used with a mandatory action (e.g., "shall ensure"), prioritize `[:IMPOSES]` over `[:DETAILS_SCOPE]`.
3.  **Hierarchy:** Use `[:INCLUDES]` ONLY for physical structure `(Regulation)-[:INCLUDES]->(Annex/Rationale/Article)` where no other legal meaning exists.
4.  **Node Centered(NEW):** Ensure full or part of text of `(Article/Annex/Rationale)` is included in Node properties, for example `Sanction.description`, `Requirement.description`, `TechCriterion.description`, not Relation properties. 
5. **Node Integrity(NEW):** Make sure properties of Node is filled as much as possible.
6. **Relevance(NEW):** For `.description` properties, ENSURE to extract only the directly relevant parts of an article, not the full-text. 
5. For `[:PERMITS]`,a chain of `()-[IMPOSES]->(Requirement)-[PERMITS]->(Stakeholder)` is allowed. 
6. For `[:MANDATED_FOR]`, `()-[IMPOSES]->(Requirement)-[MANDATED_FOR]->(Stakeholder)` is allowed. 
7. For `[:PENALIZES_WITH]`, either `(Article)-[IMPOSES]->(Requirement)-[PENALIZES_WITH]->(Sanction)-[:APPLIES_TO]->(Stakeholder)` or `()-[:PENALIZES_WITH]->(Sanction)-[:APPLIES_TO]->(Stakeholder)` is permitted. 
8. For `[:ENCOMPASSES]`, `c1.lang` and `c2.lang` should be different. 
9. For `[:SUPPLEMENTS]`, `c.lang` should not be contained in `Article/Annex/Rationale.source_data`. 
