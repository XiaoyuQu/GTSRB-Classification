from datasets import load_dataset
from torchvision import transforms
from transformers import ViTFeatureExtractor
from transformers import ViTForImageClassification, TrainingArguments, Trainer

def transform_examples(example):
    example["pixel_values"] = transform(example["image"])
    example["label"] = example["label"]
    return example
num_labels = 43
dataset = load_dataset("ilee0022/GTSRB")
feature_extractor = ViTFeatureExtractor.from_pretrained("google/vit-base-patch16-224")

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=feature_extractor.image_mean, std=feature_extractor.image_std),
])


dataset["train"] = dataset["train"].map(
    transform_examples,
    batched=False,
)
dataset["test"] = dataset["test"].map(
    transform_examples,
    batched=False,
)


model = ViTForImageClassification.from_pretrained(
    "google/vit-base-patch16-224",
    num_labels=num_labels,
    ignore_mismatched_sizes=True,
)

dataset = dataset.with_format(type="torch")

training_args = TrainingArguments(
    output_dir="./vit-finetuned-GTSRB",
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    num_train_epochs=20,
    logging_dir="./logs",
    logging_steps=10,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"],
    tokenizer=feature_extractor,
)

trainer.train()