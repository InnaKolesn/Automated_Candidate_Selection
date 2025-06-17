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
