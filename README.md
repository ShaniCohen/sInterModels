# sInterModels
This repository is the official implementation of the paper 
*"GNNs and ensemble models enhance the prediction of new sRNA-mRNA interactions in unseen conditions"*.

## Requirements

To install requirements:

```setup
pip install -r requirements.txt
```

## Data

Interactions data is provided for *Escherichia coli K12 MG1655* (NC_000913) in [Zenodo](https://zenodo.org/records/14030380)  under the `data/datasets` directory.

HT and LT datasets, including the predicted duplex, computed features and environmental conditions info.
Original source: [sInterBase](https://academic.oup.com/bioinformatics/article/39/4/btad172/7115836)

## Methods application

### Training and evaluation
```python
from models_handlers.graph_rna_model_handler import GraphRNAModelHandler
from models_handlers.xgboost_model_handler import XGBModelHandler
from models_handlers.rf_model_handler import RFModelHandler

# 1 - load data (see load_data implementation in main.py)  
data_path = "/sise/home/shanisa/GraphRNA/data"
train_fragments, test_complete, test_filtered, kwargs = load_data(data_path=data_path)
train, test = train_fragments, test_complete

# 2 - select model
model_h = GraphRNAModelHandler()  # XGBModelHandler()  # RFModelHandler()
model_args = model_h.get_model_args()

# 2 - run n-fold cross validation
cv_n_splits = 10
cv_predictions_dfs, cv_training_history = \
    model_h.run_cross_validation(X=train['X'], y=train['y'], metadata=train['metadata'], n_splits=cv_n_splits,
                                 model_args=model_args, **kwargs)
# 3 - train and test
predictions, training_history = \
    model_h.train_and_test(X_train=train['X'], y_train=train['y'], X_test=test['X'], y_test=test['y'], 
                           model_args=model_args, metadata_train=train['metadata'], metadata_test=test['metadata'], 
                           **kwargs)
test_predictions_df = predictions['out_test_pred']
```
