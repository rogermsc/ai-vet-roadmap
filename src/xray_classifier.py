```python
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc

# Configuration
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 50
NUM_CLASSES = 5  # Example: Normal, Fracture, Tumor, Arthritis, Other
SPECIES = ['canine', 'feline']  # Supported species

class VeterinaryXrayClassifier:
    """
    A multi-species veterinary X-ray classifier for common conditions
    
    This model demonstrates the technical implementation of a diagnostic
    AI system for veterinary radiographs, with species-specific adaptations.
    """
    
    def __init__(self, model_dir='./saved_models'):
        self.model_dir = model_dir
        self.models = {}
        self.preprocessing_layers = {}
        self.class_names = ['normal', 'fracture', 'tumor', 'arthritis', 'other']
        
        # Create model directory if it doesn't exist
        os.makedirs(model_dir, exist_ok=True)
        
    def build_model(self, species):
        """
        Build species-specific model architecture
        
        Args:
            species: String indicating animal species ('canine' or 'feline')
            
        Returns:
            Compiled Keras model
        """
        # Input layer
        inputs = layers.Input(shape=(*IMAGE_SIZE, 3))
        
        # Preprocessing layers - species-specific normalization
        if species == 'canine':
            # Canine-specific preprocessing
            x = layers.Rescaling(1./255)(inputs)
            # Additional canine-specific augmentation for training
            data_augmentation = tf.keras.Sequential([
                layers.RandomFlip("horizontal"),
                layers.RandomRotation(0.1),
                layers.RandomZoom(0.2),
            ])
            x = data_augmentation(x)
        elif species == 'feline':
            # Feline-specific preprocessing
            x = layers.Rescaling(1./255)(inputs)
            # Felines typically have different bone density and contrast needs
            # Additional feline-specific augmentation for training
            data_augmentation = tf.keras.Sequential([
                layers.RandomFlip("horizontal"),
                layers.RandomRotation(0.15),
                layers.RandomZoom(0.1),
                layers.RandomContrast(0.2),  # More contrast variation for feline images
            ])
            x = data_augmentation(x)
        else:
            raise ValueError(f"Unsupported species: {species}")
            
        self.preprocessing_layers[species] = x
        
        # Base model - using transfer learning from EfficientNetB0
        base_model = tf.keras.applications.EfficientNetB0(
            include_top=False,
            weights='imagenet',
            input_tensor=x
        )
        
        # Freeze base model layers for initial training
        base_model.trainable = False
        
        # Species-specific adaptation layers
        if species == 'canine':
            # Canines have more size/shape variation, so add more capacity
            x = layers.GlobalAveragePooling2D()(base_model.output)
            x = layers.Dense(256, activation='relu')(x)
            x = layers.Dropout(0.5)(x)
            x = layers.Dense(128, activation='relu')(x)
        elif species == 'feline':
            # Felines have less variation, so simpler adaptation
            x = layers.GlobalAveragePooling2D()(base_model.output)
            x = layers.Dense(128, activation='relu')(x)
            x = layers.Dropout(0.3)(x)
        
        # Common classification head
        outputs = layers.Dense(NUM_CLASSES, activation='softmax')(x)
        
        # Create model
        model = models.Model(inputs, outputs)
        
        # Compile model
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            loss='categorical_crossentropy',
            metrics=['accuracy', tf.keras.metrics.AUC(), tf.keras.metrics.Precision(), tf.keras.metrics.Recall()]
        )
        
        return model
    
    def train(self, train_data_dir, species, validation_split=0.2):
        """
        Train a species-specific model
        
        Args:
            train_data_dir: Directory containing training images organized by class
            species: Animal species for this model ('canine' or 'feline')
            validation_split: Fraction of data to use for validation
            
        Returns:
            Training history
        """
        if species not in SPECIES:
            raise ValueError(f"Unsupported species: {species}. Must be one of {SPECIES}")
        
        # Create data generators
        train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
            validation_split=validation_split
        )
        
        train_generator = train_datagen.flow_from_directory(
            os.path.join(train_data_dir, species),
            target_size=IMAGE_SIZE,
            batch_size=BATCH_SIZE,
            class_mode='categorical',
            subset='training'
        )
        
        validation_generator = train_datagen.flow_from_directory(
            os.path.join(train_data_dir, species),
            target_size=IMAGE_SIZE,
            batch_size=BATCH_SIZE,
            class_mode='categorical',
            subset='validation'
        )
        
        # Update class names from directory structure
        self.class_names = list(train_generator.class_indices.keys())
        
        # Build or load model
        if species in self.models:
            model = self.models[species]
        else:
            model = self.build_model(species)
            self.models[species] = model
        
        # Callbacks
        callbacks = [
            tf.keras.callbacks.ModelCheckpoint(
                filepath=os.path.join(self.model_dir, f'{species}_best_model.h5'),
                monitor='val_accuracy',
                save_best_only=True
            ),
            tf.keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=10,
                restore_best_weights=True
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.2,
                patience=5
            )
        ]
        
        # Train model
        history = model.fit(
            train_generator,
            epochs=EPOCHS,
            validation_data=validation_generator,
            callbacks=callbacks
        )
        
        # Save final model
        model.save(os.path.join(self.model_dir, f'{species}_final_model.h5'))
        
        # Fine-tuning phase - unfreeze some layers of the base model
        base_model = model.layers[2]  # EfficientNetB0 is the 3rd layer
        base_model.trainable = True
        
        # Freeze all layers except the last 30
        for layer in base_model.layers[:-30]:
            layer.trainable = False
            
        # Recompile with lower learning rate
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
            loss='categorical_crossentropy',
            metrics=['accuracy', tf.keras.metrics.AUC(), tf.keras.metrics.Precision(), tf.keras.metrics.Recall()]
        )
        
        # Continue training with fine-tuning
        fine_tune_history = model.fit(
            train_generator,
            epochs=EPOCHS // 2,  # Fewer epochs for fine-tuning
            validation_data=validation_generator,
            callbacks=callbacks
        )
        
        # Save fine-tuned model
        model.save(os.path.join(self.model_dir, f'{species}_fine_tuned_model.h5'))
        
        # Combine histories
        combined_history = {}
        for k in history.history.keys():
            combined_history[k] = history.history[k] + fine_tune_history.history[k]
            
        return combined_history
    
    def evaluate(self, test_data_dir, species):
        """
        Evaluate model performance on test data
        
        Args:
            test_data_dir: Directory containing test images organized by class
            species: Animal species for this model ('canine' or 'feline')
            
        Returns:
            Evaluation metrics
        """
        if species not in self.models:
            raise ValueError(f"No trained model found for species: {species}")
        
        model = self.models[species]
        
        # Create test data generator
        test_datagen = tf.keras.preprocessing.image.ImageDataGenerator()
        
        test_generator = test_datagen.flow_from_directory(
            os.path.join(test_data_dir, species),
            target_size=IMAGE_SIZE,
            batch_size=BATCH_SIZE,
            class_mode='categorical',
            shuffle=False
        )
        
        # Evaluate model
        evaluation = model.evaluate(test_generator)
        metrics = dict(zip(model.metrics_names, evaluation))
        
        # Get predictions for detailed metrics
        predictions = model.predict(test_generator)
        y_pred = np.argmax(predictions, axis=1)
        y_true = test_generator.classes
        
        # Generate classification report
        report = classification_report(
            y_true, 
            y_pred, 
            target_names=self.class_names,
            output_dict=True
        )
        
        # Generate confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        
        # Generate ROC curves and AUC for each class
        roc_data = {}
        for i, class_name in enumerate(self.class_names):
            fpr, tpr, _ = roc_curve((y_true == i).astype(int), predictions[:, i])
            roc_auc = auc(fpr, tpr)
            roc_data[class_name] = {
                'fpr': fpr.tolist(),
                'tpr': tpr.tolist(),
                'auc': roc_auc
            }
        
        # Combine all evaluation data
        evaluation_results = {
            'metrics': metrics,
            'classification_report': report,
            'confusion_matrix': cm.tolist(),
            'roc_data': roc_data
        }
        
        return evaluation_results
    
    def predict(self, image_path, species):
        """
        Make prediction on a single image
        
        Args:
            image_path: Path to the image file
            species: Animal species for this image ('canine' or 'feline')
            
        Returns:
            Dictionary with prediction results and confidence scores
        """
        if species not in self.models:
            raise ValueError(f"No trained model found for species: {species}")
        
        model = self.models[species]
        
        # Load and preprocess image
        img = tf.keras.preprocessing.image.load_img(
            image_path, 
            target_size=IMAGE_SIZE
        )
        img_array = tf.keras.preprocessing.image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        
        # Make prediction
        predictions = model.predict(img_array)
        
        # Get top prediction and confidence
        top_prediction_idx = np.argmax(predictions[0])
        top_prediction = self.class_names[top_prediction_idx]
        confidence = float(predictions[0][top_prediction_idx])
        
        # Get confidence for all classes
        all_confidences = {
            self.class_names[i]: float(predictions[0][i]) 
            for i in range(len(self.class_names))
        }
        
        # Calculate uncertainty
        entropy = -np.sum(predictions[0] * np.log2(predictions[0] + 1e-10))
        max_entropy = np.log2(len(self.class_names))
        uncertainty = entropy / max_entropy
        
        result = {
            'prediction': top_prediction,
            'confidence': confidence,
            'all_confidences': all_confidences,
            'uncertainty': uncertainty
        }
        
        return result
    
    def load_models(self):
        """Load all available trained models"""
        for species in SPECIES:
            model_path = os.path.join(self.model_dir, f'{species}_fine_tuned_model.h5')
            if os.path.exists(model_path):
                self.models[species] = tf.keras.models.load_model(model_path)
                print(f"Loaded model for {species}")
            else:
                print(f"No trained model found for {species}")
    
    def visualize_results(self, image_path, species, output_path=None):
        """
        Visualize model predictions with heatmap
        
        Args:
            image_path: Path to the image file
            species: Animal species for this image
            output_path: Path to save visualization (optional)
            
        Returns:
            Matplotlib figure
        """
        if species not in self.models:
            raise ValueError(f"No trained model found for species: {species}")
        
        # Load image
        img = tf.keras.preprocessing.image.load_img(
            image_path, 
            target_size=IMAGE_SIZE
        )
        img_array = tf.keras.preprocessing.image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        
        # Get prediction
        prediction_result = self.predict(image_path, species)
        
        # Create Grad-CAM visualization
        model = self.models[species]
        
        # Get the last convolutional layer
        last_conv_layer = None
        for layer in reversed(model.layers):
            if isinstance(layer, layers.Conv2D):
                last_conv_layer = layer.name
                break
        
        if last_conv_layer is None:
            raise ValueError("Could not find convolutional layer for Grad-CAM")
        
        # Create Grad-CAM model
        grad_model = tf.keras.models.Model(
            inputs=[model.inputs],
            outputs=[
                model.get_layer(last_conv_layer).output,
                model.output
            ]
        )
        
        # Get predicted class index
        class_idx = list(prediction_result['all_confidences'].values()).index(prediction_result['confidence'])
        
        # Record operations for automatic differentiation
        with tf.GradientTape() as tape:
            conv_outputs, predictions = grad_model(img_array)
            loss = predictions[:, class_idx]
        
        # Extract gradients
        grads = tape.gradient(loss, conv_outputs)
        
        # Compute importance weights
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
        
        # Weight feature maps with gradients
        conv_outputs = conv_outputs[0]
        heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
        heatmap = tf.squeeze(heatmap)
        
        # Normalize heatmap
        heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
        heatmap = heatmap.numpy()
        
        # Resize heatmap to original image size
        heatmap = np.uint8(255 * heatmap)
        heatmap = tf.keras.preprocessing.image.array_to_img(np.expand_dims(heatmap, axis=-1))
        heatmap = heatmap.resize((img_array.shape[2], img_array.shape[1]))
        heatmap = tf.keras.preprocessing.image.img_to_array(heatmap)
        
        # Create superimposed visualization
        superimposed_img = img_array[0] * 0.7 + heatmap * 0.3
        superimposed_img = tf.keras.preprocessing.image.array_to_img(superimposed_img)
        
        # Create figure for visualization
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
        
        # Original image with prediction
        ax1.imshow(img)
        ax1.set_title(f"Prediction: {prediction_result['prediction']}\nConfidence: {prediction_result['confidence']:.2f}")
        ax1.axis('off')
        
        # Heatmap visualization
        ax2.imshow(superimposed_img)
        ax2.set_title("Activation Heatmap")
        ax2.axis('off')
        
        # Add confidence bars for all classes
        confidences = prediction_result['all_confidences']
        sorted_confidences = sorted(confidences.items(), key=lambda x: x[1], reverse=True)
        
        # Create a new figure for confidence bars
        fig2, ax3 = plt.subplots(figsize=(10, 5))
        
        classes = [item[0] for item in sorted_confidences]
        values = [item[1] for item in sorted_confidences]
        
        bars = ax3.barh(classes, values, color='skyblue')
        ax3.set_xlim(0, 1)
        ax3.set_xlabel('Confidence')
        ax3.set_title('Prediction Confidence by Class')
        
        # Add values to bars
        for bar in bars:
            width = bar.get_width()
            ax3.text(width + 0.01, bar.get_y() + bar.get_height()/2, f'{width:.2f}', 
                    ha='left', va='center')
        
        plt.tight_layout()
        
        # Save if output path provided
        if output_path:
            fig.savefig(f"{output_path}_heatmap.png", bbox_inches='tight')
            fig2.savefig(f"{output_path}_confidences.png", bbox_inches='tight')
        
        return fig, fig2

# Example usage
if __name__ == "__main__":
    # This is a demonstration script showing how to use the classifier
    
    # Initialize classifier
    classifier = VeterinaryXrayClassifier()
    
    # Example paths - these would be replaced with actual paths in production
    train_data_dir = "./data/train"
    test_data_dir = "./data/test"
    example_image = "./data/examples/canine_radiograph.jpg"
    
    # Train models for each species
    # Note: In a real scenario, you would have species-specific datasets
    for species in SPECIES:
        print(f"Training model for {species}...")
        # Uncomment to actually train:
        # history = classifier.train(train_data_dir, species)
        
    # Load pre-trained models
    # classifier.load_models()
    
    # Make prediction on example image
    # prediction = classifier.predict(example_image, 'canine')
    # print(f"Prediction: {prediction}")
    
    # Visualize results
    # classifier.visualize_results(example_image, 'canine', './output/example_visualization')
    
    print("Script completed successfully")
```
