import tensorflow as tf
import tensorflow_text as tf_text

# Now load your model
reloaded = tf.saved_model.load('translator')


inputs = [
    'Hello', # "His room is a mess"
]



_ = reloaded.translate(tf.constant(inputs)) #warmup


result = reloaded.translate(tf.constant(inputs))

print(result[0].numpy().decode())
print()
