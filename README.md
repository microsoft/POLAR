# Project

This code base contains experiments for the paper "Automatic Calibration and Error Correction for Large Language Models via Pareto Optimal Self-Supervision"

### Dataset
The experiments are tied to the WRENCH dataset https://github.com/JieyuZ2/wrench
Please download the data in https://drive.google.com/drive/folders/1VFJeVCvckD5-qAd5Sdln4k4zJoryiEun 
It is recommended to save the data under the folder named 'wrench' in this directory.


### Code Description
pareto_learning.py implements Pareto optimal learning that trains harmonizer on multiple sources.
nlp_models.py contains harmonizer models to be trained
harmonizer_analysis.py performs harmonizer-based analysis (prediction, POLAR score calibration, source reweighting...)
rule_explanations.py contains functions to compile evidences from the supervision functions to be used in dynamic self-supervision.
                    Note that this ONLY works for the CDR dataset in WRENCH https://github.com/JieyuZ2/wrench


### Experiments and Results
To reproduce the results in the paper. Follow the two steps:

    1. Run the script run_experiments.py. This performs all the necessary computations and GPT queries for analysis.
    
    2. Run the notebook paper_results.ipynb. This notebook produces all the results in the paper.
    
    3. Notebook tutorial.ipynb contains instruction using an example dataset.


### Paper Abstract
Large language models (LLMs) have demonstrated remarkable capabilities out of box for a wide range of applications, yet accuracy still remains a major growth area, especially in mission-critical domains such as biomedicine. An effective method to calibrate the confidence level on LLM responses is essential to automatically detect errors and facilitate human-in-the-loop verification. An important source of calibration signals stems from expert-stipulated programmatic supervision, which is often available at low cost but has its own limitations such as noise and coverage. In this paper, we introduce a Pareto optimal self-supervision framework that can leverage available programmatic supervision to systematically calibrate LLM responses by producing a risk score for every response, without any additional manual efforts. This is accomplished by learning a harmonizer model to align LLM output with other available supervision sources, which would assign higher risk scores to more uncertain LLM responses and facilitate error correction. Experiments on standard relation extraction tasks in biomedical and general domains demonstrate the promise of this approach, with our proposed risk scores highly correlated with the real error rate of LLMs. For the most uncertain test instances, dynamic prompting based on our proposed risk scores results in significant accuracy improvement for off-the-shelf LLMs, boosting GPT-3 results past state-of-the-art (SOTA) weak supervision and GPT-4 results past SOTA supervised results on challenging evaluation datasets.

## Contributing

This project welcomes contributions and suggestions.  Most contributions require you to agree to a
Contributor License Agreement (CLA) declaring that you have the right to, and actually do, grant us
the rights to use your contribution. For details, visit https://cla.opensource.microsoft.com.

When you submit a pull request, a CLA bot will automatically determine whether you need to provide
a CLA and decorate the PR appropriately (e.g., status check, comment). Simply follow the instructions
provided by the bot. You will only need to do this once across all repos using our CLA.

This project has adopted the [Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct/).
For more information see the [Code of Conduct FAQ](https://opensource.microsoft.com/codeofconduct/faq/) or
contact [opencode@microsoft.com](mailto:opencode@microsoft.com) with any additional questions or comments.

## Trademarks

This project may contain trademarks or logos for projects, products, or services. Authorized use of Microsoft 
trademarks or logos is subject to and must follow 
[Microsoft's Trademark & Brand Guidelines](https://www.microsoft.com/en-us/legal/intellectualproperty/trademarks/usage/general).
Use of Microsoft trademarks or logos in modified versions of this project must not cause confusion or imply Microsoft sponsorship.
Any use of third-party trademarks or logos are subject to those third-party's policies.
