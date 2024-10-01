import os
import random
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout, Input
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
from src.constants import allowed_units  # Ensure to import allowed units from constants.py

def clean_entity_value(value):
    try:
        clean_value = value.replace('[', '').replace(']', '').replace(',', '').strip()
        parts = clean_value.split()
        if len(parts) == 2:
            return parts[0], parts[1]
        else:
            return None, None
    except Exception as e:
        print(f"Error cleaning value {value}: {e}")
        return None, None

def load_data():
    train_df = pd.read_csv(r"C:\Users\Nikku\Downloads\New folder (2)\dataset\train.csv")
    train_df[['entity_number', 'entity_unit']] = train_df['entity_value'].apply(
        lambda x: pd.Series(clean_entity_value(x))
    )
    train_df = train_df.dropna(subset=['entity_number', 'entity_unit'])
    train_df['entity_number'] = train_df['entity_number'].astype(float)
    label_encoder = LabelEncoder()
    train_df['entity_unit_encoded'] = label_encoder.fit_transform(train_df['entity_unit'])
    train_df, val_df = train_test_split(train_df, test_size=0.2, random_state=42)
    num_classes = len(label_encoder.classes_)
    return train_df, val_df, label_encoder, num_classes

def create_tf_dataset(df, image_dir, batch_size, num_classes, is_train=True):
    # Check if the image directory exists
    assert os.path.exists(image_dir), f"Image directory {image_dir} does not exist."

    datagen = ImageDataGenerator(rescale=1./255)
    generator = datagen.flow_from_dataframe(
        df,
        directory=image_dir,
        x_col='image_link',
        y_col=['entity_number', 'entity_unit_encoded'],
        batch_size=batch_size,
        class_mode='raw',
        shuffle=is_train,
        target_size=(224, 224)
    )

    def tf_generator():
        while True:
            x_batch, y_batch = next(generator)
            # Handle empty batches
            if x_batch.shape[0] == 0:
                continue
            y_entity_number = y_batch[:, 0]
            y_entity_unit = y_batch[:, 1].astype(int)
            y_entity_unit_one_hot = to_categorical(y_entity_unit, num_classes=num_classes)
            yield x_batch, {'entity_number': y_entity_number, 'entity_unit': y_entity_unit_one_hot}

    output_signature = (
        tf.TensorSpec(shape=(None, 224, 224, 3), dtype=tf.float32),
        {
            'entity_number': tf.TensorSpec(shape=(None,), dtype=tf.float32),
            'entity_unit': tf.TensorSpec(shape=(None, num_classes), dtype=tf.float32)
        }
    )

    return tf.data.Dataset.from_generator(tf_generator, output_signature=output_signature)

def build_model(num_classes):
    base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    base_model.trainable = False 

    inputs = Input(shape=(224, 224, 3))
    x = base_model(inputs, training=False)
    x = GlobalAveragePooling2D()(x)
    x = Dense(1024, activation='relu')(x)
    x = Dropout(0.5)(x)
    x = Dense(512, activation='relu')(x)

    entity_number_output = Dense(1, activation='linear', name='entity_number')(x)
    entity_unit_output = Dense(num_classes, activation='softmax', name='entity_unit')(x)

    model = Model(inputs=inputs, outputs=[entity_number_output, entity_unit_output])
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss={
            'entity_number': 'mean_squared_error', 
            'entity_unit': 'categorical_crossentropy'
        },
        metrics={
            'entity_number': 'mae', 
            'entity_unit': 'accuracy'
        }
    )
    
    return model

def main():
    train_df, val_df, label_encoder, num_classes = load_data()
    train_dataset = create_tf_dataset(train_df, r'C:\Users\Nikku\Downloads\New folder (2)\images', 32, num_classes, is_train=True)
    val_dataset = create_tf_dataset(val_df, r'C:\Users\Nikku\Downloads\New folder (2)\images', 32, num_classes, is_train=False)
    
    model = build_model(num_classes)
    model.summary()

    # Train the model
    history = model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=10,
        steps_per_epoch=len(train_df) // 32,
        validation_steps=len(val_df) // 32
    )

    model.save('entity_value_extraction_model.h5')

    # Load test dataset
    test_df = pd.read_csv(r'C:\Users\Nikku\Downloads\New folder (2)\dataset\test.csv')
    test_datagen = ImageDataGenerator(rescale=1./255)
    test_generator = test_datagen.flow_from_dataframe(
        test_df,
        directory=r'C:\Users\Nikku\Downloads\New folder (2)\images',
        x_col='image_link',
        target_size=(224, 224),
        batch_size=32,
        class_mode=None,
        shuffle=False
    )

    # Make predictions
    predictions = model.predict(test_generator)
    entity_number_pred = predictions[0].flatten()
    entity_unit_pred = predictions[1]
    entity_unit_pred_decoded = label_encoder.inverse_transform(np.argmax(entity_unit_pred, axis=1))

    # Prepare the output DataFrame
    output = pd.DataFrame({
        'index': test_df['index'],
        'prediction': [
            f"{num:.2f} {unit}" if unit in allowed_units else ""  # Check against allowed units
            for num, unit in zip(entity_number_pred, entity_unit_pred_decoded)
        ]
    })
    
    # Save output to CSV
    output.to_csv('test_out.csv', index=False)
    print("Predictions saved to test_out.csv")

if __name__ == "__main__":
    main()
