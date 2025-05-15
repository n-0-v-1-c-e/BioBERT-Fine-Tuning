# CDR Relation Extraction with BioBERT

This repository contains scripts to fine-tune BioBERT on the BC5-CDR dataset for chemical–disease relation extraction. Three variants are provided:

1. **baseline\_bc5\_finetune.py** – Simple prompt-based input with `[CHEM]`/`[DIS]` tokens.
2. **bc5\_entity.py** – Entity-marker variant inserting `[E1]…[/E1]` and `[E2]…[/E2]` around chemical and disease spans.
3. **bc5\_entity\_with\_reg\_cosine.py** – Enhanced variant adding dropout, label smoothing, and cosine LR scheduling.

---

## 📁 Repository Structure

```
BioBERT-Fine-Tuning/
├── LICENSE
├── README.md
├── his_logs
│   ├── log_baseline.log
│   ├── log_entity.log
│   └── log_entity_cosine.log
├── model
│   └── dmis-lab
│       └── biobert-v1.1
├── process_data
│   ├── dev.json
│   ├── test.json
│   └── train.json
└── script
    ├── baseline_bc5_finetune.py
    ├── bc5_entity.py
    ├── bc5_entity_with_reg_cosine.py 
    └── preprocess.py       
```

## 🚀 Requirements

* Python 3.8+
* PyTorch 1.12+
* Transformers 4.21+
* Datasets 2.x
* scikit-learn

Install via:

```bash
pip install torch torchvision transformers datasets scikit-learn
```

## 🔧 Preparation
- I provide my preprocessed data for simplicity https://huggingface.co/datasets/n0v1cee/BC5-CDR-Balanced/tree/main. Or if you would like to download and do preprocessing from beginning, try:
- Download BC5-CDR annotations. https://github.com/JHnlp/BioCreative-V-CDR-Corpus
- Run `script/preprocess.py`, remember to substitute your data path in this script, it should
  - Convert to JSON format with fields: `text`, `chemical`, `disease`, `label`. 
  - Place files under `processed_data/` as `train.json`, `dev.json`, `test.json`. 
- Download model under `model/dmis-lab/` as `biobert-v1.1` https://huggingface.co/dmis-lab/biobert-v1.1/tree/main


## 📋 Usage

  We ran training on 4 4090 GPUs using `torchrun`:

* **Baseline**

  ```bash
  torchrun --nproc_per_node=4 baseline_bc5_finetune.py
  ```

* **Entity Marker**
  ```bash
  torchrun --nproc_per_node=4 bc5_entity.py
  ```

* **Enhanced (reg + cosine)**

  ```bash
  torchrun --nproc_per_node=4 bc5_entity_with_reg_cosine.py
  ```

Scripts will:
- Load `dmis-lab/biobert-v1.1`.  
- Fine-tune on train set, evaluate on dev each epoch.  
- Save best checkpoint by `eval_f1`.  
- Finally evaluate on test set and print metrics.

## ⚙️ Configuration
Edit script-level parameters at the top of each `.py` file:
- `model_name` – pretrained model identifier.  
- Data paths (`processed_data/train.json`, etc.).  
- Hyperparameters (`learning_rate`, `batch_size`, `num_train_epochs`, etc.).

## 📊 Results
After training, look under `./result/...` for:
- `checkpoint-*` folders with saved models.  
- `log/` directory with TensorBoard logs.  

Run:
```bash
python -c "from transformers import Trainer; print('See results printed at end of script.')"
````

---

## ✨ Notes
* Check `metric_for_best_model='eval_f1'` matches your `compute_metrics` keys.
* Use `model.eval()` and reload best checkpoint before final test evaluation.

## 📜 License

MIT
