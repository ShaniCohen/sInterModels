# sInterModels
This repository is the official implementation of the paper 
*"GNNs and ensemble models enhance the prediction of new sRNA-mRNA interactions in unseen conditions"*.

## Requirements

To install requirements:

```setup
pip install -r requirements.txt
```

## Data

Interactions data (HT and LT) is provided for *Escherichia coli K12 MG1655* (NC_000913) in [Zenodo](https://zenodo.org/records/14030380)  under the `data/datasets` directory.

Original data source: [sInterBase](https://academic.oup.com/bioinformatics/article/39/4/btad172/7115836)

## Methods application

### Training and evaluation
```python
from models_handlers.kgraphrna_model_handler import kGraphRNAModelHandler
from models_handlers_old.graph_rna_model_handler import GraphRNAModelHandler
from models_handlers_old.xgboost_model_handler import XGBModelHandler
from models_handlers_old.rf_model_handler import RFModelHandler

# ----- configuration
data_path = "/sise/home/shanisa/GraphRNA/data"
data_path_2 = "/sise/home/shanisa/data_sInterModels"

# ----- load data -----
train, test_complete, test_filtered, kwargs = load_data(data_path=data_path)
# Note: these are small example datasets demonstrating the E2E flow
rna_data, train_example, test_example = load_rna_and_inter_data(data_path=data_path_2)

# ----- GraphRNA -----
graph_rna = GraphRNAModelHandler()
test = test_complete
test_predictions = train_and_evaluate(model_h=graph_rna, train=train, test=test, **kwargs)

# ----- kGraphRNA -----
k_graph_rna = kGraphRNAModelHandler()
test_predictions = train_and_test(model_h=k_graph_rna, train=train_example, test=test_example)

# ----- Decision Forests - sInterRF and sInterXGB -----
# -- sInterRF
rf = RFModelHandler()
test = test_filtered
test_predictions = train_and_evaluate(model_h=rf, train=train, test=test, **kwargs)
# -- sInterXGB
xgb = XGBModelHandler()
test = test_filtered
test_predictions = train_and_evaluate(model_h=xgb, train=train, test=test, **kwargs)
