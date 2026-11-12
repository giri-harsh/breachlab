# Data and model licences

| Asset | Licence |
|---|---|
| Source code (this repository), model weights and model card (`Pseudo-Harsh/breachlab-guard-v1`) | Apache-2.0 (see `LICENSE`) |
| BreachLab guard dataset (v1 and v2, Hugging Face `Pseudo-Harsh/breachlab-data`, `DATASET_CARD.md`) | **CC-BY-SA-3.0** |

## Why the dataset is CC-BY-SA-3.0
The training data mixes public sources (table in `DATASET_CARD.md`). It includes rows derived from `databricks/databricks-dolly-15k`, which is CC-BY-SA-3.0 and requires share-alike, so the dataset as a whole is released under CC-BY-SA-3.0. Rows from MIT and Apache-2.0 sources keep their attribution requirements; LLM-written rows are subject to the generating providers' terms. Sources that were non-commercial (toxic-chat, Alpaca) were dropped, and no row comes from the gated `allenai/wildjailbreak` (ADR-0013, ADR-0019).

## Models trained on it
The model is released under Apache-2.0 (the base models, Qwen3-0.6B and Qwen3-1.7B, are Apache-2.0). **Note:** its training data includes CC-BY-SA-3.0 rows. Whether share-alike reaches model weights trained on such text is not settled law; this project states the fact and licenses the weights Apache-2.0 as decided by the owner (ADR-0100). Anyone who needs a stricter reading should treat the weights as derived from CC-BY-SA-3.0 text.

## Not covered
Live player prompts are never part of the public dataset. The Llama Prompt Guard baseline is used only through an API and is not redistributed.
