# -*- coding: utf-8 -*-
"""
Created on Tue Dec 26 19:18:30 2017

@author: lidong
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import tensorflow as tf

#from tensorflow.python.tools import inspect_checkpoint as ic
#a=ic.print_tensors_in_checkpoint_file(file_name="D:\\RAN\\log\\model.ckpt-277227",all_tensors=True,tensor_name=None)


import argparse
import sys

import numpy as np

from tensorflow.python import pywrap_tensorflow
from tensorflow.python.platform import app
from tensorflow.python.platform import flags

from input_fn import *
import model as whole_model
# How often to record tensorboard summaries.
SUMMARY_INTERVAL = 40

# How often to run a batch through the validation model.
VAL_INTERVAL = 200

# How often to save a model checkpoint
SAVE_INTERVAL = 2000 


IMG_WIDTH = np.ceil(1242/4).astype('int32')

IMG_HEIGHT =np.ceil(375/4).astype('int32')


def print_tensors_in_checkpoint_file(file_name, tensor_name, all_tensors):
    varlist=[]
    reader = pywrap_tensorflow.NewCheckpointReader(file_name)
    if all_tensors:
      var_to_shape_map = reader.get_variable_to_shape_map()
      for key in sorted(var_to_shape_map):
        #print("tensor_name: ", key)
        varlist.append(key)
#        print(reader.get_tensor(key))
        #varlist.append(reader.get_tensor(key))
    elif not tensor_name:
      print(reader.debug_string().decode("utf-8"))
    else:
      print("tensor_name: ", tensor_name)
      print(reader.get_tensor(tensor_name))
    return varlist
images,disparities,name=get_input(0) 
#tf.device('/gpu:0')
#get input data
model=whole_model.E2EModel(images,disparities,'train')
model.build_graph()    
check= "D:\\RAN\\log\\model.ckpt-277227"   
a=print_tensors_in_checkpoint_file(file_name="D:\\RAN\\log\\model.ckpt-277227",all_tensors=True,tensor_name=None)
b= tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
saver=tf.train.Saver(b[:len(a)])
sess=tf.Session()
saver.restore(sess, "D:\\RAN\\log\\model.ckpt-277227"  )
print(sess.run(model.loss))
#c=sess.run(b)
#d=sess.run(tf.GraphKeys.GLOBAL_VARIABLES)






















