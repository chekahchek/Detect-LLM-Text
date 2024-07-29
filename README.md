# Detect-LLM-Text
Project for NTU AI6127 - Deep Neural Networks for NLP

# Introduction
In this project, both encoder and decoder based models are fine-tuned to detect presence of AI-generated text. Encoder-based models used include ELECTRA and DeBERTa while decoder-based models used is Mistral-7B-Instruct. Models were trained and evaluated on Nvidia T4 instances. 

# Running the code
You can either run train_encoder.py or train_decoder.py. The arguments accepted are:
```
Arguments:
--train_domain           The dataset category used for training. Accepts one of the following 'essay', 'reuter', 'wp'. 
--test_domain            The dataset category used for testing. Accepts one of the following 'essay', 'reuter', 'wp'.
--model                  Huggingface model to be finetuned. These models have been tested: 'mistralai/Mistral-7B-Instruct-v0.2', 'google/electra-base-discriminator', and 'microsoft/deberta-v3-base'
--max_len                Max length for tokenization. Set this to 512
--num_labels             Number of labels in the dataset. Set this to 2
--lr                     Learning Rate for fine-tuning. Set this to 2e-5 for Encoder models and 5e-5 for Decoder model
--epochs                 Number of epochs used for training. 
--train_batch_size       Training batch size
--eval_batch_size        Inference batch size   
```
