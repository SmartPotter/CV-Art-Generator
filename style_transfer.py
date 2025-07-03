import tensorflow as tf
from tensorflow.keras.applications import vgg19 # type: ignore
from tensorflow.keras.preprocessing import image as kimage # type: ignore
from tensorflow.keras import backend as K # type: ignore
import numpy as np

CONTENT_WEIGHT = 1e3
STYLE_WEIGHT = 1e-2

# Layers to use
CONTENT_LAYER = 'block5_conv2'
STYLE_LAYERS = [
    'block1_conv1',
    'block2_conv1',
    'block3_conv1',
    'block4_conv1',
    'block5_conv1'
]

def preprocess_image(image_path, target_size=(400, 400)):
    img = kimage.load_img(image_path, target_size=target_size)
    img = kimage.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    return vgg19.preprocess_input(img)

def deprocess_image(x):
    x = x.reshape((x.shape[1], x.shape[2], 3))
    x[:, :, 0] += 103.939
    x[:, :, 1] += 116.779
    x[:, :, 2] += 123.68
    x = x[:, :, ::-1]
    return np.clip(x, 0, 255).astype('uint8')

def gram_matrix(x):
    x = tf.squeeze(x)
    features = tf.reshape(x, (-1, x.shape[-1]))
    gram = tf.matmul(features, features, transpose_a=True)
    return gram

def compute_loss(model, loss_weights, init_image, gram_style_features, content_features):
    input_tensor = tf.concat([init_image], axis=0)
    features = model(input_tensor)

    loss = tf.zeros(shape=())

    content_output = features[CONTENT_LAYER]
    content_loss = tf.reduce_mean(tf.square(content_output - content_features[CONTENT_LAYER]))
    loss += loss_weights[0] * content_loss

    style_score = 0
    for layer in STYLE_LAYERS:
        style_output = features[layer]
        gram_comb = gram_matrix(style_output)
        gram_style = gram_style_features[layer]
        style_score += tf.reduce_mean(tf.square(gram_comb - gram_style))

    style_score /= float(len(STYLE_LAYERS))
    loss += loss_weights[1] * style_score

    return loss

@tf.function
def compute_grads(cfg):
    with tf.GradientTape() as tape:
        all_loss = compute_loss(**cfg)
    return tape.gradient(all_loss, cfg['init_image']), all_loss

def get_feature_representations(model, content_path, style_path):
    content_image = preprocess_image(content_path)
    style_image = preprocess_image(style_path)

    content_outputs = model(tf.constant(content_image))
    style_outputs = model(tf.constant(style_image))

    content_features = {CONTENT_LAYER: content_outputs[CONTENT_LAYER]}
    style_features = {layer: gram_matrix(style_outputs[layer]) for layer in STYLE_LAYERS}

    return content_features, style_features

def get_model():
    input_tensor = tf.keras.Input(shape=(400, 400, 3))
    vgg = vgg19.VGG19(weights='imagenet', include_top=False, input_tensor=input_tensor)
    vgg.trainable = False
    outputs = {layer.name: layer.output for layer in vgg.layers if layer.name in STYLE_LAYERS + [CONTENT_LAYER]}
    return tf.keras.Model([vgg.input], outputs)

def run_style_transfer(content_path, style_path, iterations=20, output_prefix=None):
    model = get_model()
    content_features, style_features = get_feature_representations(model, content_path, style_path)

    init_image = tf.Variable(preprocess_image(content_path), dtype=tf.float32)
    opt = tf.optimizers.Adam(learning_rate=5.0)

    loss_weights = (CONTENT_WEIGHT, STYLE_WEIGHT)

    cfg = {
        'model': model,
        'loss_weights': loss_weights,
        'init_image': init_image,
        'gram_style_features': style_features,
        'content_features': content_features
    }

    for i in range(iterations):
        grads, loss = compute_grads(cfg)
        opt.apply_gradients([(grads, init_image)])
        clipped = tf.clip_by_value(init_image, -127.5, 127.5)
        init_image.assign(clipped)

    result = init_image.numpy()
    return deprocess_image(result)
