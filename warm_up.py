
import tensorflow as tf

with tf.Graph().as_default() as g:

    # Add code that will calculate and output the Fibonacci sequence
    # using TF. You will need to make use of tf.matmul() and
    # tf.assign() to perform the multiplications and assign the result
    # back to the variable fib_seq.

    fib_matrix = tf.constant([[0.0, 1.0],
                              [1.0, 1.0]])

    ### SOLUTION START ###
    # Put your solution code here.

    # Step 1.
    # Change this line to initialize fib_seq to a 2x1 TensorFlow
    # tensor *Variable* with the initial values of 0.0 and 1.0. Hint:
    # You'll need to make sure you specify a 2D tensor of shape 2x1,
    # not a 1D tensor. See fib_matrix above (a 2x2 2D tensor) to guide
    # you.
    # Stores first two values of the fibonacci sequence
    fib_sequence = tf.Variable([[0.0],[1.0]])
    
    # Step 2.
    # Change this line to multiply fib_matrix and fib_sequence using tf.matmul()
    next_fib = tf.matmul(fib_matrix, fib_sequence)#None
    
    # Step 3.
    # Change this line to assign the result back to fib_sequence using tf.assign()
    assign_op = tf.assign(fib_sequence, next_fib)#None
    
    ### SOLUTION END ###
    
    init = tf.initialize_all_variables()
    with tf.Session() as sess:
        sess.run(init)
        for step in xrange(10):
            sess.run(assign_op)
            print(sess.run(fib_sequence)[1])