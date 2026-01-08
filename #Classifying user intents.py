#Classifying user intents


import os
print(os.environ['CONDA_DEFAULT_ENV'])

import os
from datasets import load_dataset
from sklearn.preprocessing import LabelEncoder
from setfit import SetFitModel, SetFitTrainer
from sentence_transformers.losses import CosineSimilarityLoss



# Load dataset
dataset = load_dataset('csv', data_files={
    "train": 'datasets/basic_intents_train.csv',
    "test": 'datasets/basic_intents_test.csv'
})

# Encode labels
le = LabelEncoder()
intent_dataset_train = le.fit_transform(dataset["train"]['label'])
dataset["train"] = dataset["train"].remove_columns("label").add_column("label", intent_dataset_train).cast(dataset["train"].features)

intent_dataset_test = le.fit_transform(dataset["test"]['label'])
dataset["test"] = dataset["test"].remove_columns("label").add_column("label", intent_dataset_test).cast(dataset["test"].features)




# Initialize model and trainer
model_id = "sentence-transformers/all-mpnet-base-v2"
model = SetFitModel.from_pretrained(model_id)

trainer = SetFitTrainer(
    model=model,
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"],
    loss_class=CosineSimilarityLoss,
    metric="accuracy",
    batch_size=64,
    num_iterations=20,
    num_epochs=2,
    column_mapping={"text": "text", "label": "label"}
)

# Train the model
trainer.train()

# Evaluate the model
evaluation_results = trainer.evaluate()
print("Evaluation Results:", evaluation_results)

os.makedirs('ckpt/', exist_ok=True)

trainer.model._save_pretrained(save_directory="ckpt/")


class_label_map = {
            0: "Greeting",
            1: "Farewell",
            2: "Positive Confirmation",
            3: "Negative Confirmation",
            4: "Thank you",
            5: "Book Recommendation",
            6: "Request another book",
            7: "Request book rating",
            8: "Request book pages"
         }