# 🧠 TaxoDrivenKG: Taxonomy-Driven Knowledge Graph Construction for Domain-Specific Scientific Applications

This repository implements a taxonomy-guided framework that integrates domain-specific taxonomies, large language models (LLMs), and retrieval-augmented generation (RAG) to construct high-quality knowledge graphs (KGs) from scientific literature. While our case study focuses on climate science, the framework is extensible to other specialized domains such as biomedicine and materials science.

> 🏆 **Accepted to Findings of ACL 2025**

---

## 🔍 Overview

Conventional KG construction methods often:
- Ignore curated taxonomies, leading to inconsistent semantics
- Rely solely on LLMs, resulting in hallucinated or misclassified entities

Our proposed solution addresses these limitations through:
- **Taxonomy-anchored information extraction**
- **Few-shot LLM prompting with predefined entity/relation types**
- **RAG-based output validation against domain-specific taxonomies**
- **Continuous expert-in-the-loop taxonomy extension**

The result is a more reliable, reproducible, and domain-aligned KG.

---

## 📂 Repository Structure

```bash
TaxoDrivenKG/
├── run.py                  # Entry point to run the pipeline
├── consts.py               # Configurations and constants
├── prompt_templates.py     # Prompt designs for LLM and RAG
├── utils.py                # General-purpose utilities
├── utils_RAG.py            # Retrieval and ranking utilities. Needs to be run before run.py
├── utils_LLM.py            # LLM interfacing functions
├── postprocessed.py        # Output refinement and formatting
├── outputs_others/         # Sample output files
├── requirements.txt        # Python dependencies
└── README.md               # This file
```

## 🚀 Getting Started

### 1. Install Dependencies

Make sure you have Python 3.8+ installed, then run:

```bash
pip install -r requirements.txt
```

Dataset can be downloaded from: [https://github.com/Jo-Pan/ClimateIE](https://github.com/Jo-Pan/ClimateIE)

Edit consts.py to configure:
- Input publications
- Taxonomy files
- LLM and RAG model settings

The index can be found in [this link](https://tuprd-my.sharepoint.com/:f:/g/personal/tuf28724_temple_edu/EsB5mOMQGrBJuRuV9vfMSd8Be9R6cYxSGWsD4sPVvmXzaw?e=XbHDT3). Please place the index folder in `outputs_others/`.

### 2. Run the RAG

Perform retrieval augmented generation with 
```bash
python utils_RAG.py
```

### 3. Run the extraction process.
To launch the taxonomy-driven KG extraction pipeline:
```bash
python run.py
```

### 4. Post-process based on taxonomy
```bash
python postprocessed.py
```


## 📘 Citation

If you use this framework or dataset, please cite our paper:

**Huitong Pan**, **Qi Zhang**, **Adamu Mustapha**, **Eduard C. Dragut**, and **Longin Jan Latecki**  
*Taxonomy-Driven Knowledge Graph Construction for Domain-Specific Scientific Applications*  
*Findings of the Association for Computational Linguistics (ACL), 2025*

```bibtex

@inproceedings{pan2025climateie,
  title     = {Taxonomy-Driven Knowledge Graph Construction for Domain-Specific Scientific Applications},
  author    = {Pan, Huitong and Zhang, Qi and Mustapha, Adamu and Dragut, Eduard C. and Latecki, Longin Jan},
  booktitle = {Findings of the Association for Computational Linguistics (ACL)},
  year      = {2025}
}

