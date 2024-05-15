import matplotlib.pyplot as plt
import numpy as np
import PIL
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.models import load_model
import pathlib
import warnings

warnings.filterwarnings("ignore")

# loading the dataset
local_dataset_path = "E:\\Users\\Ibrahim\\Desktop\\image recognition model\\flower_photos"
data_dir = pathlib.Path(local_dataset_path).with_suffix('').resolve()

# Checkpoint_path
checkpoint_path = "E:\\Users\\ibrahim\\Downloads\\training_checkpoint/cp.weights.h5"

# Specify the local path of the image to classify
# image_path = "E:\\Users\\Ibrahim\\Desktop\\image recognition model\\flower_photos\daisy\\5547758_eea9edfd54_n.jpg"
# image_path = "E:\\Users\\ibrahim\\Downloads\\592px-Red_sunflower.jpg"
# image_path = "E:\\Users\\ibrahim\\Downloads\\sunflower2.jpeg"
# image_path = "E:\\Users\\Ibrahim\\Desktop\\image recognition model\\flower_photos\\daisy\\5547758_eea9edfd54_n.jpg"
# image_path = "E:\\Users\\Ibrahim\\Desktop\\image recognition model\\flower_photos\\daisy\\5794835_d15905c7c8_n.jpg"
# image_path = "E:\\Users\\Ibrahim\\Desktop\\image recognition model\\flower_photos\\daisy\\54377391_15648e8d18.jpg"
# image_path = "E:\\Users\\Ibrahim\Desktop\\image recognition model\\flower_photos\\roses\\24781114_bc83aa811e_n.jpg"
# image_path = "E:\\Users\\Ibrahim\Desktop\\image recognition model\\flower_photos\\roses\\218630974_5646dafc63_m.jpg"
# image_path = "E:\\Users\\Ibrahim\Desktop\\image recognition model\\flower_photos\\roses\\229488796_21ac6ee16d_n.jpg"
image_path = "E:\\Users\\Ibrahim\Desktop\\image recognition model\\flower_photos\\tulips\\11746548_26b3256922_n.jpg"
# image_path = "E:\\Users\\Ibrahim\Desktop\\image recognition model\\flower_photos\\tulips\107693873_86021ac4ea_n.jpg"


# # Load images of roses
# roses = list(data_dir.glob('roses/*'))
# # Plot the first two images of roses
# plt.figure(figsize=(10, 5))
# for i in range(2):
#     plt.subplot(1, 2, i + 1)
#     image = PIL.Image.open(str(roses[i]))
#     plt.imshow(image)
#     plt.title('Rose Image {}'.format(i + 1))
#     plt.axis('off')
# plt.show()
#
# # Load images of tulips
# tulips = list(data_dir.glob('tulips/*'))
#
# # Plot the first two images of tulips
# plt.figure(figsize=(10, 5))
# for i in range(2):
#     plt.subplot(1, 2, i + 1)
#     image = PIL.Image.open(str(tulips[i]))
#     plt.imshow(image)
#     plt.title('Tulip Image {}'.format(i + 1))
#     plt.axis('off')
# plt.show()


# initial parameters
batch_size = 64
img_height = 180
img_width = 180

# splitting the batch to 20% for validation and 80% for training
train_ds = tf.keras.utils.image_dataset_from_directory(
    data_dir,
    validation_split=0.8,
    subset="training",
    seed=123,
    image_size=(img_height, img_width),
    batch_size=batch_size,
    verbose=0)

validation_ds = tf.keras.utils.image_dataset_from_directory(
    data_dir,
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=(img_height, img_width),
    batch_size=batch_size,
    verbose=0)

class_names = train_ds.class_names
print(class_names)

# sending training data to cache for better data fetching and processing
AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
val_ds = validation_ds.cache().prefetch(buffer_size=AUTOTUNE)

# ----------------------------------------------------------------------------------------------------------------------
# # Failed attempt to standardize the dataset by numpy
# # Calculate mean and standard deviation across all RGB channels for training data
# train_mean = np.mean(train_ds, axis=(0, 1, 2))
# train_std = np.std(train_ds, axis=(0, 1, 2))
# # Standardize training and validation data
# train_data_standardized = (train_ds - train_mean) / train_std
# val_data_standardized = (val_ds - train_mean) / train_std
# Assuming train_ds contains your training data after standardization
# # Display the first item in train_ds
# first_item = train_ds[0]
# # Print or visualize the first item
# print("First item in train_ds after standardization:")
# print(first_item)
# ----------------------------------------------------------------------------------------------------------------------


# Trying to standardize the dataset by keras & dataset.map
normalization_layer = layers.Rescaling(1. / 255)
normalized_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
image_batch, labels_batch = next(iter(normalized_ds))
first_image = image_batch[0]

# the pixel values are now in `[0,1]`.
print("min value for pixel ", np.min(first_image))
print("max value for pixel ", np.max(first_image))


# Define the preprocess_image function
def preprocess_image(image_path):
    img = tf.keras.utils.load_img(image_path, target_size=(img_height, img_width))
    img_array = tf.keras.utils.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)  # Create a batch
    return img_array


# Data augmentation to avoid overfitting by creating different data by manipulating  it
data_augmentation = keras.Sequential(
    [
        layers.RandomFlip("horizontal",
                          input_shape=(img_height,
                                       img_width,
                                       3)),
        layers.RandomRotation(0.1),
        layers.RandomZoom(0.1),
    ]
)

# Visualize a few augmented examples by applying data augmentation to the same image several times
plt.figure(figsize=(8, 8))
for images, _ in train_ds.take(1):
    for i in range(9):
        augmented_images = data_augmentation(images)
        ax = plt.subplot(3, 3, i + 1)
        plt.title('Augmentation sample')
        plt.imshow(augmented_images[0].numpy().astype("uint8"))
        plt.axis("off")

# creating a check point to callback later
checkpoint_path = "E:\\Users\\ibrahim\\Downloads\\training_checkpoint/cp.weights.h5"

checkpoint_callback = ModelCheckpoint(filepath=checkpoint_path,
                                      save_weights_only=True,
                                      verbose=1)


def train_model(train_ds, val_ds, checkpoint_path):
    num_classes = len(class_names)
    model = Sequential([
        layers.Rescaling(1. / 255, input_shape=(img_height, img_width, 3)),
        layers.Conv2D(16, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        layers.Conv2D(32, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        layers.Conv2D(64, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        layers.Dropout(0.2),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dense(num_classes, name="outputs")
    ])

    model.compile(optimizer='adam',
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])

    model.summary()

    iterations = 50
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=iterations,
        callbacks=[checkpoint_callback]
    )

    model.save('my_model.keras')
    model.load_weights(checkpoint_path)
    try:
        model.load_weights(checkpoint_path)
        print("Weights loaded successfully!")
    except FileNotFoundError:
        print("Checkpoint file not found. Please verify the file path.")

    # ----------------------------------------------------------------------------------------------------------------------

    # Visualize training results accuracy and loss
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    iterations_range = range(iterations)

    plt.figure(figsize=(8, 8))
    plt.subplot(1, 2, 1)
    plt.plot(iterations_range, acc, label='Training Accuracy')
    plt.plot(iterations_range, val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.title('Training and Validation Accuracy')

    plt.subplot(1, 2, 2)
    plt.plot(iterations_range, loss, label='Training Loss')
    plt.plot(iterations_range, val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss')
    plt.show()

    # ----------------------------------------------------------------------------------------------------------------------
    return history


def predict_with_model(model_path, image_path):
    model = load_model(model_path)
    preprocessed_image = preprocess_image(image_path)
    predictions = model.predict(preprocessed_image)
    predicted_class_index = np.argmax(predictions)
    class_names = ['daisy', 'dandelion', 'roses', 'sunflowers', 'tulips']
    predicted_class = class_names[predicted_class_index]
    confidence = np.max(predictions) * 100

    print(f"The image most likely belongs to {predicted_class} with a confidence of {confidence:.2f}%.")


choice = input("Enter '0' to start training the model or '1' to use the saved model for prediction: ")

if choice.lower() == '0':
    history = train_model(train_ds, val_ds, checkpoint_path)
elif choice.lower() == '1':
    predict_with_model('my_model.keras', image_path)
else:
    print("Invalid choice. Please enter '0' to train or '1' to predict.")
