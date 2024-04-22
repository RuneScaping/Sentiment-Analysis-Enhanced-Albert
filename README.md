
# Enhanced ALBERT for Sentiment Analysis

## Dataset Preparation
A tab-separated (.tsv) file called 'train.tsv' is required. This training dataset needs to be placed within a separate folder.

## How to Fine-tune
There are several parameters you need to set:
1. --data_dir: This is the directory where the data is stored.
2. --model_type: This is the model we are going to use for fine-tuning. In this case, it's '<i>albert</i>'.
3. --model_name_or_path: This is the variant of ALBERT that you want to use.
4. --output_dir: This is the path where you want to save the model.
5. --do_train: This needs to be set as we are training the model.

Example:
``` $ python run_glue.py --data_dir data --model_type albert --model_name_or_path albert-base-v2 --output_dir output --do_train ```

## Different Models Available for Use
The following models and their average performances are listed below:
(table taken from Google-research)

## Prediction
There are both Docker and Python files available for prediction. Just set the name of the folder where model files are stored, and then run the 'api.py' file.

Example: ``` $ python api.py ``` Or, you could also use: ``` from api import SentimentAnalyzer classifier = SentimentAnalyzer() print(classifier.predict('the movie was nice')) ```

## Acknowledgements
A huge thank you to HuggingFace for simplifying the implementation and also to Google for this remarkable pre-trained model. Credit to RuneScaping for further enhancements and maintenance of this project.