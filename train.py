import tensorflow as tf
from tensorflow.keras import optimizers, callbacks
from data_loader import load_data
from model import build_vgg16_model
from config import MODEL_PATH, EPOCHS

def train_model():
    """
    Trains the model in two phases:
    1. Train only the classification head (with frozen base layers).
    2. Fine-tune the top layers of the base model with a smaller learning rate.
    """
    train_data, val_data = load_data()
    input_shape = (224, 224, 3)
    num_classes = train_data.num_classes

    # Step 1: Train classification head with frozen base model
    model, _ = build_vgg16_model(input_shape, num_classes, trainable_base=False)

    model.compile(optimizer=optimizers.Adam(learning_rate=0.001),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    # Callbacks
    checkpoint = callbacks.ModelCheckpoint(MODEL_PATH, save_best_only=True, monitor='val_accuracy', mode='max')

    print("Training classification head...")
    model.fit(train_data, validation_data=val_data, epochs=int(EPOCHS*0.7), callbacks=[checkpoint])

    # Step 2: Fine-tune the last layers of the base model
    print("Fine-tuning model...")
    model, _ = build_vgg16_model(input_shape, num_classes, trainable_base=True, fine_tune_layers=4)

    model.compile(optimizer=optimizers.Adam(learning_rate=0.0001),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    model.fit(train_data, validation_data=val_data, epochs=int(EPOCHS*0.3), callbacks=[checkpoint])

    model.save(MODEL_PATH)
    print(f"Training completed. Model saved at {MODEL_PATH}")

if __name__ == "__main__":
    train_model()

