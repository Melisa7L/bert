import pandas as pd
import torch
import joblib
import re
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from torch.utils.data import Dataset

# Configuración para ejecutar en GPU si está disponible
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Usando dispositivo: {device}")

# Cargar el dataset
df = pd.read_excel('C:\\Users\\Meli9\\Downloads\\codigo - copia\\Data_Completo_Limpio.xlsx')

# Verificar si el archivo tiene datos
if df.empty:
    raise ValueError("El archivo Excel está vacío o no se pudo leer correctamente.")

# Omitimos el preprocesamiento con spaCy para optimizar velocidad
def limpiar_texto(texto):
    return re.sub(r'[^\w\s]', '', str(texto)).lower()

df['texto_noticia'] = df['texto_noticia'].apply(limpiar_texto)

# Mapear categorías a números
categorias = df['categoria'].unique()
categoria_a_numero = {cat: num for num, cat in enumerate(categorias)}
df['categoria_num'] = df['categoria'].map(categoria_a_numero)

# Dividir en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(
    df['texto_noticia'], df['categoria_num'], test_size=0.2, random_state=42, stratify=df['categoria_num']
)

# Tokenización con BERT en español
modelo_es = "dccuchile/bert-base-spanish-wwm-cased"
tokenizer = AutoTokenizer.from_pretrained(modelo_es)

# Definir dataset personalizado optimizado
class NoticiasDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=256):
        self.texts = texts.tolist()
        self.labels = labels.tolist()
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        encoding = self.tokenizer(
            self.texts[idx],
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'labels': torch.tensor(self.labels[idx], dtype=torch.long)
        }

# Crear datasets
train_dataset = NoticiasDataset(X_train, y_train, tokenizer)
test_dataset = NoticiasDataset(X_test, y_test, tokenizer)

# Cargar el modelo BERT en español
model = AutoModelForSequenceClassification.from_pretrained(modelo_es, num_labels=len(categorias))
model.to(device)

# Configurar entrenamiento optimizado
training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=3,
    per_device_train_batch_size=16,  # Aumentamos tamaño del batch
    per_device_eval_batch_size=16,  # Aumentamos tamaño del batch
    warmup_steps=100,  # Reducimos warmup para entrenar más rápido
    weight_decay=0.01,
    logging_dir='./logs',
    logging_steps=50,
    evaluation_strategy='epoch',
    save_strategy='epoch',
    fp16=True,  # Entrenamiento con precisión mixta
    lr_scheduler_type='cosine',  # Aprendizaje con reducción progresiva
    optim='adamw_torch'
)

# Crear el Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset
)

# Entrenar modelo
trainer.train()

# Evaluación
predictions = trainer.predict(test_dataset).predictions
y_pred = predictions.argmax(axis=1)

# Mostrar métricas
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred, target_names=categorias))

# Guardar modelo y tokenizer
model.save_pretrained('modelo_bert')
tokenizer.save_pretrained('modelo_bert')
joblib.dump(categoria_a_numero, 'categoria_mapeo.pkl')
