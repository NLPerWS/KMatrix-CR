# KMatrix-CR: A Flexible Conflict Resolution Toolkit for Knowledge-Enhanced Large Language Model System



## 🔧 Integration of Comprehensive Conflict Resolution Methods and Evaluation Datasets/Frameworks



### Conflict resolution method

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

### Support the rapid implementation of various conflict resolution methods



### A Multidimensional Unified Assessment of Conflict Resolution Methods





## 📓 Representative Knowledge Conflict Model/Method Integration

| **Type** |  **Model**/**Method**  |
| :------: | :--------------------: |
|    IC    |        Concord         |
|    IC    |        Factool         |
|    CM    |         COIECD         |
|    IC    |   Discern-and-Answer   |
|    IC    | Disinformation-Defense |
|    IM    |     Self-DETECTION     |
|    IM    |      Honest_Llama      |
|    IC    |          DoLa          |
|    CM    |          CD2           |





## 📄 Evaluation of multi-dimensional dataset integration

|               | **Type**   | **Construction Method** | **Scale** | **Causes of Conflict**                        |
| ------------- | ---------- | ----------------------- | --------- | --------------------------------------------- |
| ConflictQA    | CM         | 大模型生成+后校验       | 20091     | Misinformation  Conflict                      |
| CONFLICTINGQA | IC         | 大模型生成+后校验       | 238       | Misinformation  Conflict                      |
| ContraDoc     | IC         | 大模型生成+后校验       | 449       | Misinformation  Conflict                      |
| AttackODQA    | IC         | 实体替换、大模型生成    | 52189     | Misinformation  Conflict                      |
| Farm          | CM         | 大模型生成              | 1952      | Misinformation  Conflict                      |
| BlindGC       | CM         | 大模型生成              | 14923     | Misinformation  Conflict                      |
| KC            | CM         | 实体替换                | 9803      | Misinformation  Conflict                      |
| ConflictBank  | CM、IC、IM | 大模型生成+质量控制     | 55W       | Misinformation、  Temporal、Semantic Conflict |





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
model = LLmGenerator(model_path=llama_model_path) 
config = Config(dataset=dataset,
                model=model,
                metrics = ["acc"])
template = CMTemplate(config=config,conflict_method="coiecd")
result = template.run(output_path="cm_coiecd_"+llama_model_path.replace("/","_")+".json")
print(result)
```

