from transformers.models.auto import AutoModel, AutoTokenizer, AutoModelForSequenceClassification
print("downloading model")
pretrained_model = AutoModel.from_pretrained("microsoft/deberta-base")
print("downloading tokenizer")
tokenizer = AutoTokenizer.from_pretrained("microsoft/deberta-base")
tokenizer.save_pretrained("./model/pre_train/deberta-base")
pretrained_model.save_pretrained("./model/tokenizer/deberta-base")