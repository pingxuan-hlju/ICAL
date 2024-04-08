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
| drug_drug_sim_dis.txt      | drug-drug(diease)       | [Artical]([https://www.nlm.nih.gov/mesh/meshhome.html](https://www.nature.com/articles/s41467-017-00680-8))           |
| Similarity_Matrix_Drugs.csv| drug-drug(chemical)     | [CTD](https://ctdbase.org/)                                  |
| mat_drug_se.txt            | drug-side effect        | [SIDER](http://sideeffects.embl.de/)                         |
| se_seSmilirity.txt         | side effect-side effect | [Artical]([https://www.cuilab.cn/hmdd](https://www.nature.com/articles/s41467-017-00680-8))                           |

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
