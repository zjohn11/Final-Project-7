"""
===========================================
Sentiment Analysis Fine-tuning with BERT-Tiny
===========================================

Description:
    This script fine-tunes the 'prajjwal1/bert-tiny' BERT model 
    for binary sentiment classification using the IMDb dataset.

Dependencies:
    - transformers
    - datasets
    - torch
    - scikit-learn

Author:
    Zach Johnson
    zjohn11@okstate.edu

"""

import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import torch
from datasets import load_dataset, concatenate_datasets
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments, DataCollatorWithPadding
import seaborn as sns
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# loading the bert-tiny model and tokenizer
tokenizer = AutoTokenizer.from_pretrained("prajjwal1/bert-tiny")
model = AutoModelForSequenceClassification.from_pretrained("prajjwal1/bert-tiny", num_labels=2).to(device)

# loading the IMDB dataset
dataset = load_dataset("imdb")

# Shuffle the dataset (optional, but good for randomness)
dataset = dataset.shuffle(seed=42)

# Extracting the training and test data from the dataset
train_data = dataset["train"]
test_data = dataset["test"]

# Take 50% of the original test data and concatenate it with the train data
half_test_data = test_data.shuffle(seed=42).select(range(int(0.5 * len(test_data))))

# Use concatenate_datasets to combine the datasets
train_data = concatenate_datasets([train_data, half_test_data])

# The remaining 50% of the test data will be used for testing
test_data = test_data.shuffle(seed=42).select(range(int(0.5 * len(test_data)), len(test_data)))

# split the train_data into training and validation (90% for training, 10% for validation)
train_data, val_data = train_data.train_test_split(test_size=0.1, seed=42).values()

# Tokenizing the datasets
def tokenize(batch):
    return tokenizer(batch["text"], padding="max_length", truncation=True, max_length=512)

tokenized_train = train_data.map(tokenize, batched=True)
tokenized_val = val_data.map(tokenize, batched=True)
tokenized_test = test_data.map(tokenize, batched=True)

# Setting the format for the datasets
tokenized_train.set_format("torch", columns=["input_ids", "attention_mask", "label"])
tokenized_val.set_format("torch", columns=["input_ids", "attention_mask", "label"])
tokenized_test.set_format("torch", columns=["input_ids", "attention_mask", "label"])

# function to calculate model prediction accuracy
def compute_metrics(pred):
    logits, labels = pred
    predictions = np.argmax(logits, axis=-1)
    acc = accuracy_score(labels, predictions)
    return {"accuracy": acc}

# setting training arguments
training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=20,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=64,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    logging_dir="./logs",
    logging_steps=10,
)

# adding data collator
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

# initiating trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_val,
    data_collator=data_collator,
    compute_metrics=compute_metrics
)

# fine tuning the tiny model
trainer.train()

# 3. Evaluate after fine-tuning
eval_results = trainer.evaluate(tokenized_test)

# Extract and print only accuracy
accuracy = eval_results.get("eval_accuracy", "Accuracy not found")
print(f"Accuracy after fine-tuning: {accuracy}")

# saving the pretrained model
model.save_pretrained("./bert-tiny_sentiment-analysis")

## TESTING MODEL PEFORMANCE ##

# Predict on test set after training
predictions_output = trainer.predict(tokenized_test)

# Get predicted labels (argmax of logits)
y_pred = np.argmax(predictions_output.predictions, axis=1)

# Get true labels
y_true = predictions_output.label_ids

# Accuracy
acc = accuracy_score(y_true, y_pred)
print("Accuracy:", acc)

# Precision, Recall, F1 Score
precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='binary')
print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1)

# Confusion Matrix
# Define the labels
labels = ["Negative", "Positive"]
cm = confusion_matrix(y_true, y_pred)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels)
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix")
plt.show()
