---
tags:
- sentence-transformers
- sentence-similarity
- feature-extraction
- dense
- generated_from_trainer
- dataset_size:14194
- loss:MultipleNegativesRankingLoss
base_model: sentence-transformers/all-MiniLM-L6-v2
widget:
- source_sentence: анализ BAC
  sentences:
  - Am I winning yet? V3
  - ~300% thanks Tim 🍏, I’m out.
  - BAC CALLS for the win!
- source_sentence: акции AAPL
  sentences:
  - 'BIDEN C9UGHING!!


    SPY 100p 3/16'
  - The only DD we'll ever need. $LONG APPLE
  - Am I Doing This Right (?)
- source_sentence: stock hood
  sentences:
  - Which one of you is this?
  - When your 85 year old Grandpa starts using Robinhood
  - RH Foreshadowing tomorrow.
- source_sentence: META stock
  sentences:
  - LYFT Put holder unite
  - My face when I follow Jim Cramer Advise to Buy meta at 300
  - I got bored so I made a thing
- source_sentence: прогноз по PLTR
  sentences:
  - Why Kroger flat lined?
  - 'Earnings week ✅✅


    Meta calls Cooked me'
  - PLTR Loss but have faith it will get back up
pipeline_tag: sentence-similarity
library_name: sentence-transformers
---

# SentenceTransformer based on sentence-transformers/all-MiniLM-L6-v2

This is a [sentence-transformers](https://www.SBERT.net) model finetuned from [sentence-transformers/all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2). It maps inputs to a 384-dimensional dense vector space and can be used for semantic textual similarity, semantic search, paraphrase mining, classification, clustering, and more.

## Model Details

### Model Description
- **Model Type:** Sentence Transformer
- **Base model:** [sentence-transformers/all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) <!-- at revision 1110a243fdf4706b3f48f1d95db1a4f5529b4d41 -->
- **Maximum Sequence Length:** 256 tokens
- **Output Dimensionality:** 384 dimensions
- **Similarity Function:** Cosine Similarity
- **Supported Modality:** Text
<!-- - **Training Dataset:** Unknown -->
<!-- - **Language:** Unknown -->
<!-- - **License:** Unknown -->

### Model Sources

- **Documentation:** [Sentence Transformers Documentation](https://sbert.net)
- **Repository:** [Sentence Transformers on GitHub](https://github.com/huggingface/sentence-transformers)
- **Hugging Face:** [Sentence Transformers on Hugging Face](https://huggingface.co/models?library=sentence-transformers)

### Full Model Architecture

```
SentenceTransformer(
  (0): Transformer({'transformer_task': 'feature-extraction', 'modality_config': {'text': {'method': 'forward', 'method_output_name': 'last_hidden_state'}}, 'module_output_name': 'token_embeddings', 'architecture': 'BertModel'})
  (1): Pooling({'embedding_dimension': 384, 'pooling_mode': 'mean', 'include_prompt': True})
  (2): Normalize({'module_input_name': 'sentence_embedding', 'module_output_name': 'sentence_embedding'})
)
```

## Usage

### Direct Usage (Sentence Transformers)

First install the Sentence Transformers library:

```bash
pip install -U sentence-transformers
```
Then you can load this model and run inference.
```python
from sentence_transformers import SentenceTransformer

# Download from the 🤗 Hub
model = SentenceTransformer("sentence_transformers_model_id")
# Run inference
sentences = [
    'прогноз по PLTR',
    'PLTR Loss but have faith it will get back up',
    'Earnings week ✅✅\n\nMeta calls Cooked me',
]
embeddings = model.encode(sentences)
print(embeddings.shape)
# [3, 384]

# Get the similarity scores for the embeddings
similarities = model.similarity(embeddings, embeddings)
print(similarities)
# tensor([[1.0000, 0.6751, 0.0998],
#         [0.6751, 1.0000, 0.0888],
#         [0.0998, 0.0888, 1.0000]])
```
<!--
### Direct Usage (Transformers)

<details><summary>Click to see the direct usage in Transformers</summary>

</details>
-->

<!--
### Downstream Usage (Sentence Transformers)

You can finetune this model on your own dataset.

<details><summary>Click to expand</summary>

</details>
-->

<!--
### Out-of-Scope Use

*List how the model may foreseeably be misused and address what users ought not to do with the model.*
-->

<!--
## Bias, Risks and Limitations

*What are the known or foreseeable issues stemming from this model? You could also flag here known failure cases or weaknesses of the model.*
-->

<!--
### Recommendations

*What are recommendations with respect to the foreseeable issues? For example, filtering explicit content.*
-->

## Training Details

### Training Dataset

#### Unnamed Dataset

* Size: 14,194 training samples
* Columns: <code>anchor</code>, <code>positive</code>, <code>negative1</code>, and <code>negative2</code>
* Approximate statistics based on the first 100 samples:
  |          | anchor                                                                            | positive                                                                           | negative1                                                                          | negative2                                                                          |
  |:---------|:----------------------------------------------------------------------------------|:-----------------------------------------------------------------------------------|:-----------------------------------------------------------------------------------|:-----------------------------------------------------------------------------------|
  | type     | string                                                                            | string                                                                             | string                                                                             | string                                                                             |
  | modality | text                                                                              | text                                                                               | text                                                                               | text                                                                               |
  | details  | <ul><li>min: 4 tokens</li><li>mean: 10.05 tokens</li><li>max: 28 tokens</li></ul> | <ul><li>min: 8 tokens</li><li>mean: 87.92 tokens</li><li>max: 176 tokens</li></ul> | <ul><li>min: 5 tokens</li><li>mean: 95.77 tokens</li><li>max: 207 tokens</li></ul> | <ul><li>min: 9 tokens</li><li>mean: 82.98 tokens</li><li>max: 217 tokens</li></ul> |
* Samples:
  | anchor                                                                   | positive                                                                                                                                                                                                                                                                                        | negative1                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                   | negative2                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                       |
  |:-------------------------------------------------------------------------|:------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
  | <code>GME - Earnings Call is tomorrow. What are the expectations?</code> | <code>GME - Earnings Call is tomorrow. What are the expectations?<br><br>GameStop has had a downward trend over the year with a HUGE upward jump in Jan. Are there expectations that the call would be anything but negative? Thoughts? My position: 5/3 $10.5 Put buy 5/3 $8.5 Put Sell</code> | <code>Anyone paying attention to NKLA?<br><br>Stock is having volume greater than Tesla (TSLA) for two days. [https://finance.yahoo.com/quote/NKLA/](https://finance.yahoo.com/quote/NKLA/) \^ volume. as of 2:00 eastern volume is at 174,000,000+ [https://finance.yahoo.com/quote/TSLA/](https://finance.yahoo.com/quote/TSLA/) \^ volume.</code>                                                                                                                                                                        | <code>One Medical to the MOON<br><br>$ONEM is a hip new health service provider with excellent customer service, well-designed phone apps, and lots of telemedicine options. They IPO'd this year and their stock has soared. They are a comp to teladoc ($TDOC) that killed earnings and did very well. $ONEM dipped 30% last week for no fucking reason. I mean, insider lockup period expired but who fucking cares. The insiders are gonna hold out until after earnings if they know business is good. It got meme'd lower from $40 to $30 because traders are literally children nowadays. It's trading near its monthly low. This will easily trade over $40 at some point in the near future. Maybe Monday for all I know.</code>                                                                                                       |
  | <code>gme - earnings call is tomorrow. what are the expectations</code>  | <code>GME - Earnings Call is tomorrow. What are the expectations?<br><br>GameStop has had a downward trend over the year with a HUGE upward jump in Jan. Are there expectations that the call would be anything but negative? Thoughts? My position: 5/3 $10.5 Put buy 5/3 $8.5 Put Sell</code> | <code>Anyone paying attention to NKLA?<br><br>Stock is having volume greater than Tesla (TSLA) for two days. [https://finance.yahoo.com/quote/NKLA/](https://finance.yahoo.com/quote/NKLA/) \^ volume. as of 2:00 eastern volume is at 174,000,000+ [https://finance.yahoo.com/quote/TSLA/](https://finance.yahoo.com/quote/TSLA/) \^ volume.</code>                                                                                                                                                                        | <code>One Medical to the MOON<br><br>$ONEM is a hip new health service provider with excellent customer service, well-designed phone apps, and lots of telemedicine options. They IPO'd this year and their stock has soared. They are a comp to teladoc ($TDOC) that killed earnings and did very well. $ONEM dipped 30% last week for no fucking reason. I mean, insider lockup period expired but who fucking cares. The insiders are gonna hold out until after earnings if they know business is good. It got meme'd lower from $40 to $30 because traders are literally children nowadays. It's trading near its monthly low. This will easily trade over $40 at some point in the near future. Maybe Monday for all I know.</code>                                                                                                       |
  | <code>bear case for GME</code>                                           | <code>GME - Earnings Call is tomorrow. What are the expectations?<br><br>GameStop has had a downward trend over the year with a HUGE upward jump in Jan. Are there expectations that the call would be anything but negative? Thoughts? My position: 5/3 $10.5 Put buy 5/3 $8.5 Put Sell</code> | <code>My first trash penny $RIME bought @ $0.08<br><br>Christmas events in Brazil and Europe, and broader availability in BYD vehicles starting early 2025. ￼ Enhanced Financial Position The infusion of $9.5 million strengthens Algorhythm’s balance sheet, providing liquidity to support operations, invest in growth initiatives, and reduce debt. This improved financial footing positions the company to capitalize on emerging opportunities in the AI-driven technology and consumer electronics sectors.</code> | <code>$250k invested in an eToro QQQ CFD for 20 years, should I move to IBKR despite a huge tax bill?<br><br>ro says that for an open CFD it covers only the collateral, not the full value of the open position. \* The insurance also requires both an insolvency event and a qualifying event of misconduct, such as fraud or theft. It does not simply cover every insolvency. \* If my position grew from $250,000 to $2 million, the additional unrealised value would still be a contractual claim against eToro—not segregated cash. This has made me seriously question whether I should continue holding a large, long-term investment through eToro. I’m considering selling everything, paying the Italian capital-gains tax, and moving to Interactive Brokers, where I would try to buy a real, non-CFD UCITS ETF instead.</code> |
* Loss: [<code>MultipleNegativesRankingLoss</code>](https://sbert.net/docs/package_reference/sentence_transformer/losses.html#multiplenegativesrankingloss) with these parameters:
  ```json
  {
      "scale": 20.0,
      "similarity_fct": "cos_sim",
      "gather_across_devices": false,
      "directions": [
          "query_to_doc"
      ],
      "partition_mode": "joint",
      "hardness_mode": null,
      "hardness_strength": 0.0
  }
  ```

### Training Hyperparameters
#### Non-Default Hyperparameters

- `per_device_train_batch_size`: 32
- `num_train_epochs`: 2.0
- `learning_rate`: 3e-05
- `warmup_steps`: 0.1
- `optim`: adamw_torch
- `data_seed`: 42
- `batch_sampler`: no_duplicates

#### All Hyperparameters
<details><summary>Click to expand</summary>

- `per_device_train_batch_size`: 32
- `num_train_epochs`: 2.0
- `max_steps`: -1
- `learning_rate`: 3e-05
- `lr_scheduler_type`: linear
- `lr_scheduler_kwargs`: None
- `warmup_steps`: 0.1
- `optim`: adamw_torch
- `optim_args`: None
- `weight_decay`: 0.0
- `adam_beta1`: 0.9
- `adam_beta2`: 0.999
- `adam_epsilon`: 1e-08
- `optim_target_modules`: None
- `gradient_accumulation_steps`: 1
- `average_tokens_across_devices`: True
- `max_grad_norm`: 1.0
- `label_smoothing_factor`: 0.0
- `bf16`: False
- `fp16`: False
- `bf16_full_eval`: False
- `fp16_full_eval`: False
- `tf32`: None
- `gradient_checkpointing`: False
- `gradient_checkpointing_kwargs`: None
- `torch_compile`: False
- `torch_compile_backend`: None
- `torch_compile_mode`: None
- `use_liger_kernel`: False
- `liger_kernel_config`: None
- `use_cache`: False
- `neftune_noise_alpha`: None
- `torch_empty_cache_steps`: None
- `auto_find_batch_size`: False
- `log_on_each_node`: True
- `logging_nan_inf_filter`: True
- `include_num_input_tokens_seen`: no
- `log_level`: passive
- `log_level_replica`: warning
- `disable_tqdm`: False
- `project`: huggingface
- `trackio_space_id`: None
- `trackio_bucket_id`: None
- `trackio_static_space_id`: None
- `per_device_eval_batch_size`: 8
- `prediction_loss_only`: True
- `eval_on_start`: False
- `eval_do_concat_batches`: True
- `eval_use_gather_object`: False
- `eval_accumulation_steps`: None
- `include_for_metrics`: []
- `batch_eval_metrics`: False
- `save_only_model`: False
- `save_on_each_node`: False
- `enable_jit_checkpoint`: False
- `push_to_hub`: False
- `hub_private_repo`: None
- `hub_model_id`: None
- `hub_strategy`: every_save
- `hub_always_push`: False
- `hub_revision`: None
- `load_best_model_at_end`: False
- `ignore_data_skip`: False
- `restore_callback_states_from_checkpoint`: False
- `full_determinism`: False
- `seed`: 42
- `data_seed`: 42
- `use_cpu`: False
- `accelerator_config`: {'split_batches': False, 'dispatch_batches': None, 'even_batches': True, 'use_seedable_sampler': True, 'non_blocking': False, 'gradient_accumulation_kwargs': None}
- `parallelism_config`: None
- `dataloader_drop_last`: False
- `dataloader_num_workers`: 0
- `dataloader_pin_memory`: True
- `dataloader_persistent_workers`: False
- `dataloader_prefetch_factor`: None
- `dataloader_multiprocessing_context`: None
- `dataloader_in_order`: True
- `remove_unused_columns`: True
- `label_names`: None
- `train_sampling_strategy`: random
- `length_column_name`: length
- `ddp_find_unused_parameters`: None
- `ddp_bucket_cap_mb`: None
- `ddp_broadcast_buffers`: False
- `ddp_static_graph`: None
- `ddp_backend`: None
- `ddp_timeout`: 1800
- `fsdp`: None
- `fsdp_config`: None
- `deepspeed`: None
- `debug`: []
- `skip_memory_metrics`: True
- `do_predict`: False
- `resume_from_checkpoint`: None
- `local_rank`: -1
- `prompts`: None
- `batch_sampler`: no_duplicates
- `multi_dataset_batch_sampler`: proportional
- `router_mapping`: {}
- `learning_rate_mapping`: {}
- `warmup_ratio`: None

</details>

### Training Logs
| Epoch  | Step | Training Loss |
|:------:|:----:|:-------------:|
| 0.1126 | 50   | 2.7339        |
| 0.2252 | 100  | 2.1988        |
| 0.3378 | 150  | 1.8888        |
| 0.4505 | 200  | 1.7346        |
| 0.5631 | 250  | 1.5930        |
| 0.6757 | 300  | 1.4071        |
| 0.7883 | 350  | 1.4059        |
| 0.9009 | 400  | 1.2810        |
| 1.0135 | 450  | 1.3078        |
| 1.1261 | 500  | 1.1323        |
| 1.2387 | 550  | 1.0740        |
| 1.3514 | 600  | 1.1548        |
| 1.4640 | 650  | 1.0686        |
| 1.5766 | 700  | 1.1203        |
| 1.6892 | 750  | 1.0613        |
| 1.8018 | 800  | 1.0196        |
| 1.9144 | 850  | 1.1025        |


### Training Time
- **Training**: 15.0 minutes

### Framework Versions
- Python: 3.14.5
- Sentence Transformers: 6.2.0
- Transformers: 5.17.0
- PyTorch: 2.12.0
- Accelerate: 1.15.0
- Datasets: 5.0.1
- Tokenizers: 0.23.2

## Additional Resources

- [Training and Finetuning Embedding Models with Sentence Transformers](https://huggingface.co/blog/train-sentence-transformers): the end-to-end guide for training or finetuning Sentence Transformer models.
- [Introduction to Matryoshka Embedding Models](https://huggingface.co/blog/matryoshka): variable-size embeddings that can be truncated with minimal quality loss.
- [Binary and Scalar Embedding Quantization for Significantly Faster & Cheaper Retrieval](https://huggingface.co/blog/embedding-quantization): post-training compression of embedding vectors.
- [Multimodal Embedding & Reranker Models with Sentence Transformers](https://huggingface.co/blog/multimodal-sentence-transformers): use text, image, audio, and video models through the same API.
- [Training and Finetuning Multimodal Embedding & Reranker Models with Sentence Transformers](https://huggingface.co/blog/train-multimodal-sentence-transformers): train multimodal embedding models, with a Visual Document Retrieval walkthrough.

## Citation

### BibTeX

#### Sentence Transformers
```bibtex
@inproceedings{reimers-2019-sentence-bert,
    title = "Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks",
    author = "Reimers, Nils and Gurevych, Iryna",
    booktitle = "Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing",
    month = "11",
    year = "2019",
    publisher = "Association for Computational Linguistics",
    url = "https://arxiv.org/abs/1908.10084",
}
```

#### MultipleNegativesRankingLoss
```bibtex
@misc{oord2019representationlearningcontrastivepredictive,
      title={Representation Learning with Contrastive Predictive Coding},
      author={Aaron van den Oord and Yazhe Li and Oriol Vinyals},
      year={2019},
      eprint={1807.03748},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/1807.03748},
}
```

<!--
## Glossary

*Clearly define terms in order to be accessible across audiences.*
-->

<!--
## Model Card Authors

*Lists the people who create the model card, providing recognition and accountability for the detailed work that goes into its construction.*
-->

<!--
## Model Card Contact

*Provides a way for people who have updates to the Model Card, suggestions, or questions, to contact the Model Card authors.*
-->