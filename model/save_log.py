import tensorflow as tf
from tensorflow.python.platform import gfile
import os.path
filename = 'model/classify_image_graph_def.pb'

graph = tf.get_default_graph()
if filename.endswith('.meta'):
    _ = tf.train.import_meta_graph(filename)
else:
    with gfile.FastGFile(filename, 'rb') as f:
        text = f.read()
        print (len(text))
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(text)
    tf.import_graph_def(graph_def)
summary_write = tf.summary.FileWriter(os.path.join(os.path.basename(filename), 'visualize') , graph)