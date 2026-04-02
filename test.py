from tensorflow.keras import Model
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.regularizers import l2

train_path = "C:/Users/Yonas/Desktop/files/projects/AI Plant disease detector/Dataset/Dataset_for_Tomato/Train"
validation_path = "C:/Users/Yonas/Desktop/files/projects/AI Plant disease detector/Dataset/Dataset_for_Tomato/Validate"

# ===== Strong Augmentation =====
train_gen = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    rotation_range=45,
    width_shift_range=0.2,
    height_shift_range=0.2,
    zoom_range=0.3,
    shear_range=0.3,
    horizontal_flip=True,
    vertical_flip=True,
    brightness_range=[0.6,1.4],
    fill_mode='nearest'
).flow_from_directory(
    train_path,
    target_size=(224,224),
    batch_size=64,
    class_mode='categorical'
)

val_gen = ImageDataGenerator(
    preprocessing_function=preprocess_input
).flow_from_directory(
    validation_path,
    target_size=(224,224),
    batch_size=64,
    class_mode='categorical'
)

# ===== Build Model =====
base = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224,224,3))

x = base.output
x = GlobalAveragePooling2D()(x)
x = BatchNormalization()(x)

# ===== New Dense Layers with Regularization =====
x = Dense(128, activation='relu', kernel_regularizer=l2(0.001))(x)  # L2 prevents overfitting
x = Dropout(0.5)(x)
x = Dense(64, activation='relu', kernel_regularizer=l2(0.001))(x)   # second dense layer
x = Dropout(0.5)(x)

predictions = Dense(3, activation='softmax')(x)

model = Model(inputs=base.input, outputs=predictions)

# ===== Freeze Base Layers =====
for layer in base.layers:
    layer.trainable = False

# ===== Compile Stage 1 =====
model.compile(
    optimizer=Adam(learning_rate=1e-4),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# ===== Callbacks =====
early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
checkpoint = ModelCheckpoint("best_model.h5", monitor='val_accuracy', save_best_only=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.3, patience=3, min_lr=1e-6)

# ===== Stage 1 Training =====
model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=10,
    callbacks=[early_stop, checkpoint, reduce_lr]
)

# ===== Stage 2: Fine-Tune Last 15 Layers =====
for layer in base.layers[-15:]:
    layer.trainable = True

model.compile(
    optimizer=Adam(learning_rate=5e-6),  # very low learning rate
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=10,
    callbacks=[early_stop, checkpoint, reduce_lr]
)

model.save("tomatoV2_final.h5")