Files included:
- inference.py          : PyTorch inference for ResNet variants
- pipeline_config.json  : JSON specifying data paths and model choice
- model_params.toml     : TOML with per-architecture training params
- pipeline.py           : Pipeline integrating JSON + TOML and training loop
- grid_search.json      : JSON defining grid search combinations
- grid_search.py        : Runner for the grid search
