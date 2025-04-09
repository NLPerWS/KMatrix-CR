# KMatrix-CR: A Flexible Conflict Resolution Toolkit for Knowledge-Enhanced Large Language Model System



**Integration of Comprehensive Conflict Resolution Methods and Evaluation Datasets/Frameworks：Support the rapid implementation of various conflict resolution methods and A Multidimensional Unified Assessment of Conflict Resolution Methods**



## 🔧 Conflict resolution method

- CM Conflict Resolution
  - Faithful to Context
  - Faithful to Memory
  - Disentangling Sources
  - Improving Factuality

- IC Conflict Resolution
  - Eliminating Conflict
  - Improving Robustness

- IM Conflict Resolution
  - Improving Consistency
  - Improving Factuality



## 📓 Representative Knowledge Conflict Model/Method Integration

| **Type** |  **Model**/**Method**  |
| :------: | :--------------------: |
|    IC    |        ICL-whole         |
|    IC    |        ICL-seprate         |
|    IC    |        Factool         |
|    CM    |         COIECD         |
|    CM    |         Context-Faithful         |
|    CM    |         Aware-Decoding         |
|    CM    |         Retrieveorgenerated         |
|    CM    |         Llms_believe_the_earth_is_flat         |
|    IM    |        Dola          |
|    IM    |        Concord          |




## 📄 Evaluation of multi-dimensional dataset integration

|               | **Type**   | **Construction Method**            | **Scale** | **Causes of Conflict**                        |
| ------------- | ---------- | ---------------------------------- | --------- | --------------------------------------------- |
| ConflictQA    | CM         | LLM generation+Post-validation     | 20091     | Misinformation  Conflict                      |
| CONFLICTINGQA | IC         | LLM generation+Post-validation     | 238       | Misinformation  Conflict                      |
| ContraDoc     | IC         | LLM generation+Post-validation     | 449       | Misinformation  Conflict                      |
| AttackODQA    | IC         | Entity Replacement、LLM generation | 52189     | Misinformation  Conflict                      |
| Farm          | CM         | LLM generation                     | 1952      | Misinformation  Conflict                      |
| BlindGC       | CM         | LLM generation                     | 14923     | Misinformation  Conflict                      |
| KC            | CM         | Entity Replacement                 | 9803      | Misinformation  Conflict                      |
| ConflictBank  | CM、IC、IM | LLM generation+Quality Control     | 55W       | Misinformation、  Temporal、Semantic Conflict |



## 📄 Knowledgeconflictresolutionevaluation

|                        |     Method     |  Acc   |
| :--------------------: | :------------: | :----: |
| CM Conflict Resolution | w/o knowledge  | 14.69% |
|                        |  w/ knowledge  | 28.59% |
|                        |    +COIECD     | 66.44% |
| IC Conflict Resolution | w/o knowledge  | 0.01%  |
|                        |  w/ knowledge  | 42.70% |
|                        | +Discriminator | 50.00% |



## 💫 Example of Tool Usage/Operation

Evaluation Metrics Integration (accuracy, F1-score, and additional intermediate metrics to assess the degree of conflict resolution)

```python
from kmatrix_cr.config.config import Config
from kmatrix_cr.dataset.dataset import Dataset
from kmatrix_cr.generator.llm_generator import LLmGenerator
from kmatrix_cr.generator.openai_generator import OpenAiGenerator
from kmatrix_cr.template.template_cm import CMTemplate
from kmatrix_cr.template.template_ic import ICTemplate
from kmatrix_cr.template.template_im import IMTemplate

llama_model_path = "meta-llama/Llama-2-7b-chat-hf"
dataset = Dataset(dataset_path="nq.jsonl")
llm_model = LLmGenerator(model_path=llama_model_path) 
config = Config(dataset=dataset,
                llm_model=llm_model,
                metrics = ["acc"])
template = CMTemplate(config=config,conflict_method="coiecd")
result = template.run(output_path="cm_coiecd_"+llama_model_path.replace("/","_")+".json")
```

