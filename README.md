# Automated_Candidate_Selection

Automated CV scoring based on averaging the predictions of multiple ranking models

## Structure 

- 'models/' - pretrained pipelines in .dill format
- 'src/score/' - models loading scipts, pipelines' decriptions and utils
  
## Installation in Collab
```python
!pip install dill sentence-transformers scikit-learn umap-learn hdbscan catboost
!git clone https://github.com/<your_user>/my_resume_scorer.git
import sys
sys.path.append("/content/my_resume_scorer/src")
