from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import os

# Allow CPU usage and avoid duplicate OpenMP issue
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# Step 1: Load dataset
train_path = "PlantVillage"
img_size = 64
batch_size = 32

train_datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

train_gen = train_datagen.flow_from_directory(
    train_path,
    target_size=(img_size, img_size),
    batch_size=batch_size,
    class_mode='categorical',
    subset='training'
)

val_gen = train_datagen.flow_from_directory(
    train_path,
    target_size=(img_size, img_size),
    batch_size=batch_size,
    class_mode='categorical',
    subset='validation'
)

# Step 2: Load or create model
if os.path.exists("crop_model.h5"):
    print("🔁 Loading existing model...")
    model = load_model("crop_model.h5")
    initial_epoch = 20  # 👈 Change this to the last completed epoch
else:
    print("🆕 Creating new model...")
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(img_size, img_size, 3)),
        MaxPooling2D(2, 2),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D(2, 2),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(train_gen.num_classes, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    initial_epoch = 0

# Step 3: Callbacks
os.makedirs('saved_models', exist_ok=True)

checkpoint = ModelCheckpoint(
    filepath='saved_models/crop_epoch_{epoch:02d}_val_acc_{val_accuracy:.2f}.h5',
    monitor='val_accuracy',
    save_best_only=False,
    save_weights_only=False,
    verbose=1
)

early_stop = EarlyStopping(patience=10, restore_best_weights=True)

# Step 4: Train (resume from initial_epoch)
history = model.fit(
    train_gen,
    epochs=30,  # 👈 Final target epoch (you’ll train till 30 here)
    initial_epoch=initial_epoch,
    validation_data=val_gen,
    callbacks=[checkpoint, early_stop]
)

# Step 5: Save final model
model.save("crop_model.h5")
print("✅ Final model saved as crop_model.h5")
