# ICAL

## Introduction

Interactive multi-hypergraph inferring and channelenhanced and attribute-enhanced learning for drug-related side effect predictio”).

# File

```markdown				
-data : data set										
-main : model training and test
```

## Dataset

| File_name                  | Data_type               | Source                                                       |
| -------------------------- | ----------------------- | ------------------------------------------------------------ |
| drug_drug_sim_dis.txt      | drug-drug(diease)       | [MeSH](https://www.nlm.nih.gov/mesh/meshhome.html)           |
| Similarity_Matrix_Drugs.csv| drug-drug(chemical)     | [CTD](https://ctdbase.org/)                                  |
| mat_drug_se.txt            | drug-side effect        | [SIDER](http://sideeffects.embl.de/)                         |
| se_seSmilirity.txt         | side effect-side effect | [HMDD](https://www.cuilab.cn/hmdd)                           |

## Environment

```markdown
packages:
python == 3.9.0
torch == 1.13.0
numpy == 1.23.5
scikit-learn == 1.2.2
scipy == 1.10.1
pandas == 2.0.1
matplotlib == 3.7.1
```

# Run

```python
python ./main.py
```
