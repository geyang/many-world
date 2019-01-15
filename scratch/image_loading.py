import tensorflow as tf
from PIL import Image

# filename_queue = tf.train.string_input_producer(['../figures/PointMass-v0.png'])  # list of files to read
# reader = tf.WholeFileReader()
# key, value = reader.read(filename_queue)

string_tensor = tf.convert_to_tensor(['../figures/PointMass-v0.png'])
filename_queue = tf.data.Dataset.from_tensor_slices(string_tensor).shuffle(string_tensor.shape[0]).repeat(10)


# Reads an image from a file, decodes it into a dense tensor, and resizes it
# to a fixed shape.
def _parse_function(filename, label):
    image_string = tf.read_file(filename)
    image_decoded = tf.image.decode_jpeg(image_string)
    image_resized = tf.image.resize_images(image_decoded, [28, 28])
    return image_resized, label


# A vector of filenames.
filenames = tf.constant(['../figures/PointMass-v0.png'])

# `labels[i]` is the label for the image in `filenames[i].
labels = tf.constant([37])

dataset = tf.data.Dataset.from_tensor_slices((filenames, labels))
dataset = dataset.map(_parse_function)

init_op = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init_op)

    for img, label in dataset:
        pass

    dataset = tf.data.Dataset.range(100)
    iterator = dataset.make_one_shot_iterator()
    next_element = iterator.get_next()

    for i in range(100):
        value = sess.run(next_element)
        assert i == value



# my_img = tf.image.decode_png(value)  # use png or jpg decoder based on your files.
#
# init_op = tf.global_variables_initializer()
# with tf.Session() as sess:
#     sess.run(init_op)
#
#     # Start populating the filename queue.
#     coord = tf.train.Coordinator()
#     threads = tf.train.start_queue_runners(coord=coord)
#
#     for i in range(1):  # length of your filename list
#         image = my_img.eval()  # here is your image Tensor :)
#
#     Image.fromarray((image * 255).reshape(image.shape[:2]).astype("uint8"), mode="L").show()
#
#     coord.request_stop()
#     coord.join(threads)
