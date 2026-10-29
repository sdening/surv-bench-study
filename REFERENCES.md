# References & acknowledgments

## Core Survival Analysis Libraries and Reference Code Implementations

- survhive  
  https://github.com/compbiomed-unito/survhive

- scikit-survival  
  https://scikit-survival.readthedocs.io

- pycox  
  https://github.com/havakv/pycox

- coxkan  
  https://github.com/knottwill/CoxKAN

- SurvivalEVAL  
  https://github.com/shi-ang/SurvivalEVAL

- survival-intro
  https://github.com/georgehc/survival-intro

- xgboost  
  https://xgboost.readthedocs.io

- xgbse  
  https://github.com/loft-br/xgboost-survival-embeddings

- survtrace  
  https://github.com/RyanWangZf/SurvTRACE

---

## Dataset Resources

- SurvSet  
  https://github.com/ErikinBC/SurvSet/tree/main

- FinSurvival  
  https://github.com/Large-Transaction-Models/DMLR_DeFi_Survival_Benchmark

- Additional preprocessing-specific references can be found in: [dataset configurations](https://github.com/sdening/surv-bench-study/tree/main/Data_Preprocessing/configs)

---

`Model_Training/third_party/survtrace/` contains a locally adapted copy of [SurvTRACE](https://github.com/RyanWangZf/SurvTRACE). Adaptations are marked with `#manual edit`.

---

# BibTeX References

```bibtex
@misc{birolo2025survhivepackageconsistentlyaccess,
    title={SurvHive: a package to consistently access multiple survival-analysis packages}, 
    author={Giovanni Birolo and Ivan Rossi and Flavio Sartori and Cesare Rollo and Tiziana Sanavia and Piero Fariselli},
    year={2025},
    eprint={2502.02223},
    archivePrefix={arXiv},
    primaryClass={q-bio.QM},
    url={https://arxiv.org/abs/2502.02223}, 
}

@article{Polsterl2020ScikitSurvival,
  author  = {P{\"o}lsterl, Sebastian},
  title   = {scikit-survival: A Library for Time-to-Event Analysis Built on Top of scikit-learn},
  journal = {Journal of Machine Learning Research},
  volume  = {21},
  number  = {212},
  pages   = {1--6},
  year    = {2020},
  url     = {https://www.jmlr.org/papers/v21/20-729.html}
}

@software{kvamme_pycox_2019,
  author = {Kvamme, Håvard},
  title = {Time-to-event prediction with PyTorch},
  year = {2019},
  url = {https://github.com/havakv/pycox},
note   = {Accessed: 2026-03-13}
}


@software{coxkan_github_2025,
  author = {Knottenbelt, William},
  title = {CoxKAN: Kolmogorov-Arnold Networks for Survival Analysis},
  year = {2025},
  url = {https://github.com/knottwill/CoxKAN},
note   = {Accessed: 2026-03-16}
}

@article{qi2024survivaleval,
year = {2024},
month = {01},
pages = {453-457},
title = {{SurvivalEVAL}: A Comprehensive Open-Source Python Package for Evaluating Individual Survival Distributions},
author={Qi, Shi-ang and Sun, Weijie and Greiner, Russell},
volume = {2},
journal = {Proceedings of the AAAI Symposium Series},
doi = {10.1609/aaaiss.v2i1.27713}
}

@misc{chen2024introductiondeepsurvivalanalysis,
      title={An Introduction to Deep Survival Analysis Models for Predicting Time-to-Event Outcomes}, 
      author={George H. Chen},
      year={2024},
      eprint={2410.01086},
      archivePrefix={arXiv},
      primaryClass={stat.ML},
      url={https://arxiv.org/abs/2410.01086}, 
}

@online{xgboosting_aft_survival,
  author = {{xgboost developers}},
  title  = {Survival Analysis with Accelerated Failure Time},
  year   = {2026},
  url    = {https://xgboost.readthedocs.io/en/latest/tutorials/aft_survival_analysis.html},
  note   = {Accessed: 2026-05-06}
}

@software{xgbse2020github,
  author = {Davi Vieira and Gabriel Gimenez and Guilherme Marmerola and Vitor Estima},
  title = {XGBoost Survival Embeddings: improving statistical properties of XGBoost survival analysis implementation},
  url = {http://github.com/loft-br/xgboost-survival-embeddings},
  version = {0.3.3},
  year = {2021}
}

@article{survtrace_2022_wang,
  author       = {Zifeng Wang and
                  Jimeng Sun},
  title        = {SurvTRACE: Transformers for Survival Analysis with Competing Events},
  journal      = {CoRR},
  volume       = {abs/2110.00855},
  year         = {2021},
  url          = {https://arxiv.org/abs/2110.00855},
  eprinttype    = {arXiv},
  eprint       = {2110.00855},
  timestamp    = {Tue, 29 Aug 2023 16:24:44 +0200},
  biburl       = {https://dblp.org/rec/journals/corr/abs-2110-00855.bib},
  bibsource    = {dblp computer science bibliography, https://dblp.org}
}
```

## Dataset-specific references

Additional source URLs recorded in dataset configurations and preprocessing scripts:

- **ADNI_dem_biom_hipp.yaml**: [Source 1](http://dx.doi.org/10.1007/978-3-030-43823-4_37)
- **borrow_repay.yaml**: [Source 1](https://github.com/Large-Transaction-Models/DMLR_DeFi_Survival_Benchmark/blob/main/dataPreprocessing.R), [Source 2](https://arxiv.org/abs/2507.14160)
- **mimic_iv.yaml**: [Source 1](https://physionet.org/content/mimiciv/3.1/), [Source 2](https://doi.org/10.13026/kpb9-mt58), [Source 3](https://arxiv.org/abs/2008.08637)
- **support2.yaml**: [Source 1](https://github.com/georgehc/survival-intro/blob/main/datasets.py)
- **create_borrow_repay_csv.R**: [Source 1](https://github.com/Large-Transaction-Models/DMLR_DeFi_Survival_Benchmark)
- **run_preprocess_kkbox.py**: [Source 1](https://github.com/georgehc/survival-kernets/blob/main/datasets.py)
- **run_preprocess_mimic_iv.py**: [Source 1](https://arxiv.org/abs/2008.08637), [Source 2](https://www.sciencedirect.com/science/article/pii/S1532046418300716)

Acceptance at NeurIPS 2026, Evaluations & Datasets Track.