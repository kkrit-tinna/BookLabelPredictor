# readme.md 

This folder contains the final consolidated code for the BookLabelPredictor Project by Nihar Baddigam, Yifei Shi and Ying-Jen Chiang, and contains contributions from all of them.


Summary of included files:
- main.py -- handles the running of code, allows user interaction to select the model they want run
- models.py -- contains code for the models used in this project
- metrics.py -- contains evaluation functions
- preprocessing.py -- contains the preprocessing pipeline shared across all models
- runners.py -- contains runners for each model, used by main. Parameters for individual models like number of epochs can be changed here
- generate_embeddings.py -- contains code to generate 384 dimension SBERT embeddings
- requirements.txt -- contains module requirements to run this code

--- To Run ---

1. Clone the repository from the link in the Project Report(https://github.com/kkrit-tinna/BookLabelPredictor). 
2. Ensure that all the above files are present in the same folder.
3. Download the dataset from this link: https://www.kaggle.com/datasets/elvinrustam/books-dataset/code 
4. Place the downloaded "BooksDataset.csv" file in the same folder as the code. 
5. Open a command prompt terminal from the project's root directory
6. In the prompt, enter: "python main.py --model "<MODEL NAME>"" where <MODEL NAME> is one of: logreg_tfidf, cosine_sbert, nn_frozen, nn_unfrozen, prototype
	a) "logreg_tfidf" runs the baseline logistic regression model. It prints to the terminal which labels it is excluding, i.e. not making predictions on because it hasn't encountered 		them in the train set, before printing evaluation metrics
	b) "cosine_sbert" runs the direct cosine similarity comparison between generated SBERT description and label embeddings, printing metrics to the terminal
	c) "nn_frozen" runs the neural network projection head on frozen embeddings with fixed parameters as set in runners.py, printing validation and test loss per epoch and the final 		evaluation metrics
	d) "nn_unfrozen" runs the neural network projection head on unfrozen embeddings, allowing changes to embedding space and printing validation and test loss per epoch and the final 		evaluation metrics
	e) "prototype" runs the prototype embedding model that creates label vector prototypes using description vectors in the training space

Note: When running any of b) to e) for the first time, the runner will first read the dataset from the csv before preprocessing it. It will then generate SBERT embeddings for all of the models. This process can take a while and is computationally heavy, so running with a GPU is recommended. Once completed, the same train-val-test split of the dataset will be used for all models run from main.py. This split is unique to the constant "RANDOM_STATE" in the preprocessing.py file to allow direct comparison between models; if this constant is changed from the default of 7, the code will create a new train-test split, unique to the new random state. 

