# IRT_Persuasion

Code for the Paper "An Item Response Framework for Persuasion". The paper can currently be found on arxiv:

The `irt_lib` folder contains shared code for models and data preparation. The `debates` and `editorials` folders contain dataset specific processing, results and analysis. In each subfolder, the README file contains overall details on where to download the raw data, then the `DataAssembly.ipynb` notebooks show how the data was transformed into feature+label pairs. Finally, the `Models.ipynb` notebooks contain the experiments included in the paper.

The training was conducted using Amazon SageMaker on a `ml.p3.2xlarge` machine. The `model_env.lst` contains the python requirements/library versions used during training. The `data_env.lst` file contains the python requirements for preprocessing. 

Note: For the "SpaCy" preprocessing, you will need to download the models using `python -m spacy download en_core_web_md`
