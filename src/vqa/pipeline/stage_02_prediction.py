import json
import numpy as np
import cv2
import tensorflow as tf
import gensim
from nltk.tokenize import word_tokenize
import pickle
import h5py

# Load the word2vec model
model_path = '/home/sushant/Desktop/nyu-cv-final-project/project_code/artifacts/Copy of GoogleNews-vectors-negative300.bin.gz'
model_w2v = gensim.models.KeyedVectors.load_word2vec_format(model_path, binary=True)

# Function to preprocess the question
def preprocess_question(question):
    txt = word_tokenize(question.lower())
    feat = []
    for word in txt:
        try:
            feat.append(model_w2v[word])
        except:
            pass
    feat = np.array(feat)
    if len(feat) < 21:
        padding = np.zeros((21 - len(feat), 300))
        feat = np.concatenate((feat, padding), axis=0)
    return feat.reshape((1, 21, 300))

# Function to preprocess the image
def preprocess_image(image_path):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (448, 448))
    image = np.expand_dims(image, axis=0) / 255.0

    # VGG16 model
    vgg_model = tf.keras.applications.VGG16(include_top=False, weights='imagenet', input_shape=(448, 448, 3))
    vgg_model.trainable = False
    vgg_output = vgg_model(image)

    # Dimension reduction model
    dimen_red = tf.keras.Sequential([
        tf.keras.layers.Conv2D(300, kernel_size=(1, 1), input_shape=(14, 14, 512)),
        tf.keras.layers.Reshape((196, 300)),
        tf.keras.layers.Permute((2, 1)),
        tf.keras.layers.Dense(21),
        tf.keras.layers.Permute((2, 1))
    ])

    # Reshape VGG output to match the expected input shape of dimen_red
    vgg_output_reshaped = tf.reshape(vgg_output, [-1, 14, 14, 512])
    processed_image = dimen_red(vgg_output_reshaped)
    return processed_image


# Load the trained model
model = tf.keras.models.load_model('/home/sushant/Desktop/nyu-cv-final-project/project_code/artifacts/VQA_Model')

# Function to make prediction
def predict_answer(image_path, question):
    processed_image = preprocess_image(image_path)
    processed_question = preprocess_question(question)

    # Ensure the order of inputs is correct: first the question, then the image
    prediction = model.predict([processed_question, processed_image])
    predicted_index = np.argmax(prediction)

    # Load the label to answer mapping
    label2answer_path = '/home/sushant/Desktop/nyu-cv-final-project/project_code/artifacts/cache/trainval_label2ans.pkl'
    with open(label2answer_path, 'rb') as f:
        label2answer = pickle.load(f)
    
    predicted_answer = label2answer[predicted_index]
    return predicted_answer

# Example usage
image_path = '/home/sushant/Desktop/nyu-cv-final-project/project_code/artifacts/VQA_RAD Image Folder/synpic676.jpg'
question = "Are the costophrenic angles blunted?"
answer = predict_answer(image_path, question)
print(f"Predicted Answer: {answer}")

image_path = '/home/sushant/Desktop/nyu-cv-final-project/project_code/artifacts/VQA_RAD Image Folder/synpic676.jpg'
question = "Is the heart enlarged?"
answer = predict_answer(image_path, question)
print(f"Predicted Answer: {answer}")

image_path = '/home/sushant/Desktop/nyu-cv-final-project/project_code/artifacts/VQA_RAD Image Folder/synpic28602.jpg'
question = "Is the trachea midline?"
answer = predict_answer(image_path, question)
print(f"Predicted Answer: {answer}")


