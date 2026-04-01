from tensorflow.keras import Model
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.regularizers import l2
import numpy as np
from sklearn.utils.class_weight import compute_class_weight

train_path = "C:/Users/Yonas/Desktop/files/projects/AI Plant disease detector/Dataset/Train"
validation_path = "C:/Users/Yonas/Desktop/files/projects/AI Plant disease detector/Dataset/Validate"

# ===============================
# 1️⃣ Data Augmentation
# ===============================
trainGenerator = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    rotation_range=45,
    width_shift_range=0.2,
    height_shift_range=0.2,
    zoom_range=0.3,
    shear_range=0.3,
    horizontal_flip=True,
    vertical_flip=True,
    brightness_range=[0.6, 1.4],
    channel_shift_range=30,
    fill_mode='nearest'
).flow_from_directory(
    train_path,
    target_size=(224,224),
    batch_size=64,  # bigger batch for stability
    class_mode='categorical'
)

validGenerator = ImageDataGenerator(
    preprocessing_function=preprocess_input
).flow_from_directory(
    validation_path,
    target_size=(224,224),
    batch_size=64,
    class_mode='categorical'
)

# Compute class weights to handle any imbalance
labels = trainGenerator.classes
class_weights = compute_class_weight(
    class_weight='balanced',
    classes=np.unique(labels),
    y=labels
)
class_weights = dict(enumerate(class_weights))

# ===============================
# 2️⃣ Model
# ===============================
baseModel = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224,224,3))

x = baseModel.output
x = GlobalAveragePooling2D()(x)
x = BatchNormalization()(x)
x = Dense(128, activation='relu', kernel_regularizer=l2(0.001))(x)  # L2 regularization
x = Dropout(0.5)(x)
predictLayer = Dense(3, activation='softmax')(x)

model = Model(inputs=baseModel.input, outputs=predictLayer)

# Freeze all layers first
for layer in baseModel.layers:
    layer.trainable = False

# Compile
model.compile(
    optimizer=Adam(learning_rate=0.0001),
    loss="categorical_crossentropy",
    metrics=['accuracy']
)

# ===============================
# 3️⃣ Callbacks
# ===============================
early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
checkpoint = ModelCheckpoint("best_model.h5", monitor='val_accuracy', save_best_only=True)
reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.3,
    patience=3,
    min_lr=1e-6
)

# ===============================
# 4️⃣ Stage 1: Short training
# ===============================
model.fit(
    trainGenerator,
    validation_data=validGenerator,
    epochs=10,
    callbacks=[early_stop, checkpoint, reduce_lr],
    class_weight=class_weights
)

model.save("tomatoV2.h5")
