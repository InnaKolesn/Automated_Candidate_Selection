# Automated_Candidate_Selection

Automated CV scoring based on averaging the predictions of multiple ranking models:
- Univercity Rating
- Education Level and Education Status
- Study Direction
- Experience Field
- Hard Skills
- Soft Skills
- IT Courses

## Structure 

- 'models/' - pretrained pipelines in .dill format
- 'src/score/' - models loading scipts, pipelines' decriptions and utils
  
## Installation in Collab
```python
!pip install dill sentence-transformers scikit-learn umap-learn hdbscan catboost
!git clone https://github.com/InnaKolesn/Automated_Candidate_Selection
import sys
sys.path.append("/content/Automated_Candidate_Selection/src")
