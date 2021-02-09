# in_height = 2
# in_width = 2
# strides = (1,2,2,1)
# filter_height = 3
# filter_width = 3

# from math import ceil

# out_height = ceil(float(in_height) / float(strides[1]))
# out_width  = ceil(float(in_width) / float(strides[2]))

# pad_along_height = max((out_height - 1) * strides[1] +
#                     filter_height - in_height, 0)
# pad_along_width = max((out_width - 1) * strides[2] +
#                    filter_width - in_width, 0)
# pad_top = pad_along_height // 2
# pad_bottom = pad_along_height - pad_top
# pad_left = pad_along_width // 2
# pad_right = pad_along_width - pad_left

# print(pad_top, pad_bottom, pad_left, pad_right)
# print(out_height, out_width)

# out_height = in_height * strides[1]
# out_width  = in_width * strides[2]

# padding_height = (strides[1] * (in_height - 1) + filter_height - out_height) / 2
# padding_width  = (strides[2] * (in_width - 1) + filter_width - out_width) / 2

# print(padding_height, padding_width)
# print(out_height, out_width)

# reverse engineer how tf.nn.conv2d_transpose dels with different output_shape parameters
# turns out that tensorflow cuts output at [:,x:,x:,0], where x is difference between actual output and output_shape

# run with both eval_id = 7 and eval = 8
# will return single value of resulting matrix
# to confirm compare values and see that both are same

eval_id = 8

import tensorflow as tf

tf.set_random_seed(42)

tf_tensor = tf.random.uniform((1,4,4,1))

tf_filter = tf.Variable(tf.random_normal([3, 3, 1, 1],
                                           stddev=0.01,
                                           dtype=tf.float32),
                          name='encoder_f')

tf_conv_7 = tf.nn.conv2d_transpose(
    tf_tensor, tf_filter, [1,7,7,1], [1,2,2,1], padding='SAME'
)

tf_conv_8 = tf.nn.conv2d_transpose(
    tf_tensor, tf_filter, [1,8,8,1], [1,2,2,1], padding='SAME'
)

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)

    if eval_id == 7:
        v = sess.run(tf_conv_7)
        print(v[:,:,:,0][0][0][0])
    elif eval_id == 8:
        v = sess.run(tf_conv_8)
        print(v[:,1:,1:,0][0][0][0])