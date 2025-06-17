# Automated\_Candidate\_Selection

Automated CV scoring framework that unifies multiple pretrained pipelines:

* **University Rating**
* **Education Level & Status**
* **Study Direction**
* **Experience Field**
* **Hard Skills**
* **Soft Skills**
* **IT Courses**

This repo provides:

* **`src/scorer/`**: Python modules defining reusable `Pipeline` classes and utilities.
* **`models/`**: Pretrained `.dill` artifacts for each scoring pipeline.
* **`notebooks/`**: Example notebooks demonstrating how to load models and score new resumes.

## Project Structure

```
Automated_Candidate_Selection/
├── models/                    # Сохранённые .dill модели
│   └── study_direction_cluster.dill   # пример кластеризации Study Direction
│
├── src/scorer/                # Main components soyrce code
│   ├── clusterer.py           # `Clusterer` PCA→UMAP→HDBSCAN
│   ├── config.py              # Constans and paths
│   ├── embeddings.py          #  BERT-embeddings utils (mean/max, cash)
│   ├── pipelines.py           # `StudyDirectionPipeline`, `ExperienceFieldPipeline`, etc
│   ├── ranker.py              # `CatBoostRanker`- wrapper
│   └── utils.py               # Text-preprossing
│
├── notebooks/                 
│   └── example_usage.ipynb    # How to load a model in Google Colab
│
├── README.md                  
└── requirements.txt           
```

## Installation (e.g. in Colab)

```bash
!pip install dill sentence-transformers scikit-learn umap-learn hdbscan catboost
!git clone https://github.com/InnaKolesn/Automated_Candidate_Selection.git
%cd Automated_Candidate_Selection
import sys
sys.path.append('src/scorer')
```

(e.g. in Colab)

```bash
!pip install dill sentence-transformers scikit-learn umap-learn hdbscan catboost
!git clone https://github.com/InnaKolesn/Automated_Candidate_Selection.git
%cd Automated_Candidate_Selection
import sys
sys.path.append('src/scorer')
```

## Key Modules

### `pipelines.py`

* **`StudyDirectionPipeline`**:

  * Loads two `.dill` clusterers: study and selected directions.
  * Loads a CatBoostRanker for scoring study-direction using `selected_cluster` + optional features.

* **`ExperienceFieldPipeline`**:

  * Loads a `.dill` clusterer for selected directions.
  * Aggregates candidate skills (SBERT mean/max embeddings).
  * Uses CatBoostRanker on `[selected_cluster] + additional_features` to predict HR score.

* **`CombinedScoringPipeline`**: (optional)

  * A generic template for combining multiple field-specific pipelines into a single scorer.

### `utils.py`

* Text normalization and token mapping utilities.
* CSV/Excel loader with path resolution.

## Usage Example (`notebooks/usage_example.ipynb`)

Below is a minimal snippet showing how to score new resumes:

```python
import dill
import pandas as pd
from pipelines import StudyDirectionPipeline, ExperienceFieldPipeline

# 1) Load pre-saved pipelines
sdp = dill.load(open('models/full_study_pipeline.dill','rb'))
efp = dill.load(open('models/exp_field_pipeline.dill','rb'))

# 2) Read new resume data
df = pd.read_excel('../data/resumes.xlsx')

# 3) Score Study Direction
study_scores = sdp.predict(
    selected_texts    = df['Selected_Direction'],
    study_texts       = df['Study_Direction'],
    additional_features = df[['Experience_Field_Cluster','Ed_Status_Level']]
)
df['StudyScore_Pred'] = study_scores

# 4) Score Experience Field
exp_scores = efp.predict(
    selected_texts    = df['Selected_Direction'].tolist(),
    lists_of_skills   = df['IT_Skills_list'].tolist(),
    additional_features = df[['Selected_Direction_Cluster','Ed_Status_Level']]
)
df['ExpFieldScore_Pred'] = exp_scores

# 5) Save or display
df.head()
```

Adjust paths and column names as needed.
