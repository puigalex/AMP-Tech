from __future__ import print_function
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)
import tensorflow as tf

# Parameters
taza_aprendizaje = 0.01
numero_pasos = 500
lotes = 128
display_step = 100

# Network Parameters
neuronas_capa1 = 256 # 1st layer number of neurons
neuronas_capa2 = 256 # 2nd layer number of neurons
numero_entradas = 784 # MNIST data input (img shape: 28*28)
numero_clasificaciones = 10 # MNIST total classes (0-9 digits)

# tf Graph input
X = tf.placeholder("float", [None, numero_entradas])
Y = tf.placeholder("float", [None, numero_clasificaciones])

# Store layers weight & bias
diccionario_pesos_W = \
    {
    'entrada_capa1': tf.Variable(tf.random_normal([numero_entradas, neuronas_capa1])),
    'capa1_capa2': tf.Variable(tf.random_normal([neuronas_capa1, neuronas_capa2])),
    'capa2_salida': tf.Variable(tf.random_normal([neuronas_capa2, numero_clasificaciones]))
    }
diccionario_sesgos_b = \
    {
    'biascapa1': tf.Variable(tf.random_normal([neuronas_capa1])),
    'biascapa2': tf.Variable(tf.random_normal([neuronas_capa2])),
    'biasultimacapa': tf.Variable(tf.random_normal([numero_clasificaciones]))
    }


# Create model
def neural_net(x):
    # Hidden fully connected layer with 256 neurons
    layer_1 = tf.add(tf.matmul(x, diccionario_pesos_W['entrada_capa1']), diccionario_sesgos_b['biascapa1'])
    # Hidden fully connected layer with 256 neurons
    layer_2 = tf.add(tf.matmul(layer_1, diccionario_pesos_W['capa1_capa2']), diccionario_sesgos_b['biascapa2'])
    # Output fully connected layer with a neuron for each class
    out_layer = tf.matmul(layer_2, diccionario_pesos_W['capa2_salida']) + diccionario_sesgos_b['biasultimacapa']
    return out_layer

# Construct model
logits = neural_net(X)
prediction = tf.nn.softmax(logits)

# Define loss and optimizer
loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
    logits=logits, labels=Y))
optimizer = tf.train.AdamOptimizer(learning_rate=taza_aprendizaje)
train_op = optimizer.minimize(loss_op)

# Evaluate model
correct_pred = tf.equal(tf.argmax(prediction, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# Initialize the variables (i.e. assign their default value)
init = tf.global_variables_initializer()

# Start training
with tf.Session() as sess:

    # Run the initializer
    sess.run(init)

    for step in range(1, numero_pasos + 1):
        batch_x, batch_y = mnist.train.next_batch(lotes)
        # Run optimization op (backprop)
        sess.run(train_op, feed_dict={X: batch_x, Y: batch_y})
        if step % display_step == 0 or step == 1:
            # Calculate batch loss and accuracy
            loss, acc = sess.run([loss_op, accuracy], feed_dict={X: batch_x,
                                                                 Y: batch_y})
            print("Step " + str(step) + ", Minibatch Loss= " + \
                  "{:.4f}".format(loss) + ", Training Accuracy= " + \
                  "{:.3f}".format(acc))

    print("Optimization Finished!")

    # Calculate accuracy for MNIST test images
    print("Testing Accuracy:", \
        sess.run(accuracy, feed_dict={X: mnist.test.images,
                                      Y: mnist.test.labels}))