import tensorflow as tf
import numpy as np
import cv2
import os

# Define the paths to the dataset
unstabilized_path = 'path/to/unstabilized/videos'
stabilized_path = 'path/to/stabilized/videos'

# Define the hyperparameters
batch_size = 32
learning_rate = 0.001
epochs = 10

# Define the model architecture
model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3)),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dense(2, activation='linear')
])

# Define the loss function and optimizer
loss_fn = tf.keras.losses.MeanSquaredError()
optimizer = tf.keras.optimizers.Adam(learning_rate)

# Define the training dataset
unstabilized_videos = os.listdir(unstabilized_path)
stabilized_videos = os.listdir(stabilized_path)

dataset = tf.data.Dataset.from_tensor_slices((unstabilized_videos, stabilized_videos))

def preprocess_video(video_file):
    # Load the video file
    cap = cv2.VideoCapture(video_file)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()
    # Resize the frames to 128x128
    resized_frames = []
    for frame in frames:
        resized_frame = cv2.resize(frame, (128, 128))
        resized_frames.append(resized_frame)
    # Convert the frames to a numpy array and normalize the pixel values
    video_array = np.array(resized_frames) / 255.0
    return video_array

def load_video(unstabilized_file, stabilized_file):
    unstabilized_video = preprocess_video(os.path.join(unstabilized_path, unstabilized_file))
    stabilized_video = preprocess_video(os.path.join(stabilized_path, stabilized_file))
    return unstabilized_video, stabilized_video

def prepare_dataset_for_training(unstabilized_file, stabilized_file):
    unstabilized_video, stabilized_video = tf.numpy_function(load_video, [unstabilized_file, stabilized_file], [tf.float32, tf.float32])
    return unstabilized_video, stabilized_video

train_dataset = dataset.map(prepare_dataset_for_training).batch(batch_size)

# Define the training loop
@tf.function
def train_step(inputs, targets):
    with tf.GradientTape() as tape:
        predictions = model(inputs, training=True)
        loss = loss_fn(targets, predictions)
    grads = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))
    return loss

for epoch in range(epochs):
    epoch_loss = 0.0
    for batch, (inputs, targets) in enumerate(train_dataset):
        batch_loss = train_step(inputs, targets)
        epoch_loss += batch_loss
    print('Epoch {}/{} - Loss: {:.4f}'.format(epoch + 1, epochs, epoch_loss / (batch + 
