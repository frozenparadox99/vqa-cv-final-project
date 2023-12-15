# link - http://www.cs.virginia.edu/~vicente/vislang/slides/wasimonica.pdf
import os,h5py
# Image Model
import tensorflow as tf
import numpy as np
from keras import Model
import warnings
warnings.filterwarnings('ignore')

'''  Preprocessing with the VGG 16 Model  '''

model = tf.keras.applications.VGG16(include_top=False,weights='imagenet',
        input_shape=(448,448,3))
#print("The Last layer")
last_layer = model.layers[-1].output   # Last layer has an output layer of (14,14,512)
model = Model(model.input,last_layer)
model.trainable = False
# print(model.summary())
def extract_feature(image):
    ''' Preprocessing with VGG Netowrk'''
    image = model(image)
    return image  # Shape is (196,512)

'''  The below model will convert (196,512) to (21,300) (i.e same as the dimension of word embedding)  '''

dimen_red = tf.keras.Sequential()  # Use for converting (196,512) -> (21,300)
dimen_red.add(tf.keras.layers.Conv2D(300,kernel_size=(1,1),input_shape= (14,14,512)))
dimen_red.add(tf.keras.layers.Reshape((196,300)))
dimen_red.add(tf.keras.layers.Permute((2,1)))  # Reshaping about the axis, useful for applying the dense network
dimen_red.add(tf.keras.layers.Dense(21))
dimen_red.add(tf.keras.layers.Permute((2,1)))  # Reshaping about the axis, useful for applying the dense network

train_dir = r'/home/sushant/Desktop/nyu-cv-final-project/project_code/artifacts/QA/' # Containing .h5 file
images = []
ans = []
ques = []
count = 0
content = os.listdir(train_dir)[:500]   # The GPU Memory became full after this, hence had to take just these much samples :(
length = len(content)
for i in content:
    # Reading the data
    file = h5py.File(train_dir+i)
    images.append(np.array(file['.']['image_vector'][()]))
    ans.append(np.array(file['.']['answers'][()]))
    ques.append(np.array(file['.']['ques_train'][()]))
    count+=1
    if count%100 == 0:
      print("The count is:",count,"and the percentage proportion is:",round(100*count/length,2),'%')
images = tf.convert_to_tensor(np.array(images))   # For the GPU purpose
ans = tf.convert_to_tensor(np.array(ans))
ques = tf.convert_to_tensor(np.array(ques))

l = []
length = images.shape[0]
for i,j in enumerate(images):
  l.append(model(tf.reshape(j,[1,448,448,3])))  # It was not possible directly on GPU, hence had to use for loop
  if i%100 ==0:
    print("The count is:",i,"and the percentage proportion is:",round(100*i/length,2),'%')
images = tf.convert_to_tensor(np.array(l) )


l = []
length = images.shape[0]
for i,j in enumerate(images):
  l.append(dimen_red(j))     # Making it to the same shape as that of question embedding
  if i%100 ==0:
    print("The count is:",i,"and the percentage proportion is:",round(100*i/length,2),'%')
images = tf.convert_to_tensor(np.array(l))

images = tf.reshape(images,[length,21,300])
img = images  #Tensor containing images
que = ques  # Tensor containing question vector
img = img/255.0 # Normalizing

''' The below mentioned two Input objects of keras will be useful for making the model '''

ques = tf.keras.layers.Input((21,300))  # Input Model (for ques)
images = tf.keras.layers.Input((21,300)) # Input Model (for images)


from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense, Dropout, Embedding, LSTM, GlobalMaxPooling1D, SpatialDropout1D,Flatten,Concatenate, Multiply, Permute, Reshape, Bidirectional

''' Imagica is for the preprocessing of the image part'''
imagica = Dense(512,activation='tanh')(images)
imagica = Dense(512,activation='tanh')(imagica)


''' quesa is for the ques layer, which means preprocessing of the question layer'''
quesa = Bidirectional(LSTM(512, dropout = 0.3,return_sequences = True,input_shape = (21,300)))(ques)
quesa = Dense(512, activation = 'relu')(quesa)
quesa = Dropout(0.3)(quesa)
quesa = Dense(512, activation = 'relu')(quesa)
quesa = Dropout(0.3)(quesa)

attention_1 = Dense(512, activation='softmax')(quesa)
attention_1 = Reshape((512, 21))(attention_1)
attention_1 = Permute((2, 1))(attention_1)
attention_1_applied = Multiply()([imagica, attention_1])

attention_2 = Dense(512, activation='softmax')(attention_1_applied)
attention_2 = Reshape((512, 21))(attention_2)
attention_2 = Permute((2, 1))(attention_2)
attention_2_applied = Multiply()([imagica, attention_2])

''' Concatenating both image and the question layer'''
combined = Concatenate()([attention_2_applied, quesa])
combined = Flatten()(combined)

out = tf.keras.layers.Dense(476,activation='softmax')(combined) # Final output has 476 different categories, you can check by finding length of uniquue answers :)


from keras.utils import plot_model
model = Model([ques,images],[out])
plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)

model.compile(optimizer=tf.keras.optimizers.Adam(lr = 0.01),loss ='sparse_categorical_crossentropy',metrics=['accuracy'])

answers = tf.keras.utils.to_categorical(ans)

model.fit([img,que],ans,epochs = 50,batch_size=32,verbose=1)

model.save('/home/sushant/Desktop/nyu-cv-final-project/project_code/artifacts/VQA_Model')

prediction = tf.argmax(model.predict([img,que]),axis=1).numpy()

import json
x = open('/home/sushant/Desktop/nyu-cv-final-project/project_code/artifacts/trainset.json','r')
train = json.load(x)
train[0]

import pickle
x = open('/home/sushant/Desktop/nyu-cv-final-project/project_code/artifacts/cache/trainval_label2ans.pkl','rb')
a2lab = pickle.load(x)
import cv2
img_path = '/home/sushant/Desktop/nyu-cv-final-project/project_code/artifacts/VQA_RAD Image Folder/'
import matplotlib.pyplot as plt

for i in range(100,150,2):
  plt.figure(figsize=(10,8))
  image = cv2.imread(img_path+ train[i]['image_name'])
  plt.imshow(image)
  value = "Actual: "+str(train[i]['answer'])+' predicted value: '+ str(a2lab[prediction[i]])
  plt.axis('off')
  print("The Question is:",train[i]['question'])
  print("The answer is:",value)
  print("*"*50)
  plt.show()