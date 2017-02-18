import tensorflow as tf
import sys
import os

image_extensions = ['.jpg', '.png', '.jpeg', '.bmp', '.gif', '.tif', '.tiff']

# change this as you see fit
image_path = sys.argv[1]

# Read in the image_data
# image_data = tf.gfile.FastGFile(image_path, 'rb').read()

# Loads label file, strips off carriage return
label_lines = [line.rstrip() for line 
                   in tf.gfile.GFile("retrained_labels.txt")]

# Unpersists graph from file
with tf.gfile.FastGFile("retrained_graph.pb", 'rb') as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())
    _ = tf.import_graph_def(graph_def, name='')

txt_output = open(os.path.split(image_path)[0] + "/evaluation.txt", "w")

with tf.Session() as sess:

    # Feed the image_data as input to the graph and get first prediction
    softmax_tensor = sess.graph.get_tensor_by_name('final_result:0')
    
    i = 0

    for file in sorted(os.listdir(image_path)):
        
        if os.path.splitext(file)[1] in image_extensions:

            image_data = tf.gfile.FastGFile(image_path + file, 'rb').read()

            predictions = sess.run(softmax_tensor, \
                    {'DecodeJpeg/contents:0': image_data})
            
            # Sort to show labels of first prediction in order of confidence
            top_k = predictions[0].argsort()[-len(predictions[0]):][::-1]
            
            i += 1
            print ("(" + str(i) + "/" + str(len(os.listdir(image_path))-1) + ")  " + file)
            txt_output.write("\nTarget: " + file + "\n\n")

            for node_id in top_k:
                human_string = label_lines[node_id]
                score = predictions[0][node_id]
                # print('%s (score = %.5f)' % (human_string, score) + "\n")
                txt_output.write('%s (score = %.5f)' % (human_string, score) + "\n")

txt_output.write("\n")
txt_output.close()
