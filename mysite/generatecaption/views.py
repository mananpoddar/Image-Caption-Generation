from django.shortcuts import render
import tensorflow as tf
from tensorflow.python.keras.applications import VGG16
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.layers import Input, Dense, Embedding
from tensorflow.python.keras.layers.recurrent import GRU
from .models import Images,load_model
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from .utils import *



list_variables = load_model.model()
graph = list_variables[3]
vgg_model = list_variables[0]
language_model = list_variables[1]
tokenizer = list_variables[2]
VGG_last_layer = list_variables[5]
image_model = list_variables[4]






def index(request):
    if request.method == 'POST':
        image = Images(
            image = request.FILES.get('image')

        )
        image.save()
        images = request.FILES.get('image')
     
        path = "./media/images/"+str(images)
        max_tokens = 30
        global language_model,tokenizer,graph,sess,image_model,VGG_last_layer,vgg_model
        with graph.as_default():
                        
            path_checkpoint = 'model_weights.keras'
            language_model.load_weights(path_checkpoint)

            image_model = VGG16(include_top=True, weights='imagenet')
            VGG_last_layer = image_model.get_layer('fc2')
            vgg_model = Model(inputs = image_model.input, outputs = VGG_last_layer.output)
        
            cap = generate_captions(vgg_model,language_model,tokenizer,path,graph,max_tokens)
            return render(request,"generatecaption/viewImage.html",{"cap":cap})
   

    return render(request,"generatecaption/index.html")




dataDir='.'
dataType='train2017'

embedding_size = 256
cell_state_size = 512
num_words = 15000
activation_vector_length = 4096

def load_image(path,size=(224,224,)):
    img = Image.open(path)
    img = img.resize(size=size,resample=Image.LANCZOS)
    img = np.array(img)/255.0
    if (len(img.shape) == 2):
        img = np.repeat(img[:, :, np.newaxis], 3, axis=2)
    
    return img


def generate_captions(vgg_model,language_model,tokenizer,path,graph,max_tokens=30):
    

    start = 'ssstrt '
    end = ' eenddd'
    img = load_image(path)
    image_batch = np.expand_dims(img, axis=0)
    with graph.as_default():
            
        activations = vgg_model.predict(image_batch)
        lang_input = np.zeros(shape=(1,max_tokens),dtype=np.int)
        
        token_index = tokenizer.word_index[start.strip()]
        output_text = ''
        count = 0
        
        while token_index != end and count < max_tokens :
            lang_input[0,count] = token_index
            X = [np.array(activations),np.array(lang_input)]
            lang_out = language_model.predict(X)
            one_hot = lang_out[0,count,:]
            token_index = np.argmax(one_hot)
            word = tokenizer.token_to_word(token_index)
            output_text+=" "+str(word)
            count+=1
        
        print('The output is :',output_text,'.')
        
    return output_text
    # plt.imshow(img)
    # plt.show()



# Create your views here.
 # img = load_image(path)
                # image_batch = np.expand_dims(img, axis=0)
                # v = tf.Variable(image_batch, dtype=tf.float32)
                # init = tf.variables_initializer([v, ])
                # sess.run(init)
                # node = vgg_model(v)

                # sess.run(node)
            # sess = tf.Session()
            # vgg_model = sess.run(vgg_model)
            # language_model = sess.run(language_model)
            # tokenizer = sess.run(tokenizer)
                
            # y = tf.contrib.eager.py_func(func=generate_captions,inp=[vgg_model,language_model,tokenizer,path,graph,max_tokens],Tout=tf.float32)
            # with tf.Session(graph=graph) as sess:   
            #     # generate_captions(vgg_model,language_model,tokenizer,path,graph,max_tokens=30)
            #     x = sess.run([y],feed_dict={vgg_model:vgg_model,language_model:language_model,tokenizer:tokenizer,graph:graph})

              # global graph
        #with tf.Session(graph=graph) as sess:
            # sess.run(initialize)
            # sess.run(assign)
        # list_variables = model_image.model()

        # vgg_model = sess.run(vgg_model)
        # language_model = sess.run(language_model)
        # tokenizer = sess.run(tokenizer)
        # graph = sess.run(graph)
        # graph = list_variables[3]
        # vg_model = list_variables[0]
        # language_model = list_variables[1]    
        # tokenizer = list_variables[2]
# with tf.Session():
    # list_variables = model_image.model()
# list_variables = model_image.model()
# graph = list_variables[3]
# vgg_model = list_variables[0]
# language_model = list_variables[1]
# tokenizer = list_variables[2]

#     vgg_model = tf.Variable(True, use_resource=True)
#     language_model = tf.Variable(True, use_resource=True)
#     tokenizer = tf.Variable(True, use_resource=True)
#     graph = tf.Variable(True, use_resource=True)
#     initialize = tf.global_variables_initializer()
#     assign1 = vgg_model.assign(list_variables[0])
#     assign2 = language_model.assign(list_variables[1])
#     assign3 = tokenizer.assign(list_variables[2])
#     assign4 = graph.assign(list_variables[3])