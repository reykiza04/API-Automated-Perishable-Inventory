import json
import tensorflow as tf
from flask import Flask, request, Response

app = Flask(__name__)

def load_fruit_model():
    return tf.keras.models.load_model('models/model_fruit.h5')

model_fruit = load_fruit_model()

@app.route('/inference', methods=['GET', 'POST'])
def fruit():
    if request.method == 'POST':
        file = request.files['file']
        img = load_image(file.read(), (210, 210))
        clazz, prob = parse_fruit_output(model_fruit(img).numpy()[0])
        return Response(json.dumps({
            'class': clazz,
            'probability': prob,
            }),
            status=200, mimetype='application/json')
    return """
    <form method="POST" enctype="multipart/form-data" action='inference'>
    <input type='file' name='file' />
    <input type='submit' value='Submit' />
    </form>
    """


def load_image(bytes,size):
    img = tf.image.decode_image(bytes, channels=3, expand_animations=False)
    img = tf.image.resize(img, [*size])
    img = tf.image.convert_image_dtype(img, tf.float32)
    img = tf.expand_dims(img, axis=0)
    return img / 255.0

def parse_fruit_output(output):
    config ={
        0: 'Fresh Products',
        1: 'Fresh Products',
        2: 'Fresh Products',
        3: 'Rotten Products',
        4: 'Rotten Products',
        5: 'Rotten Products'
        }
    return config[output.argmax()] , int(output.max() * 100)


# app.run(port=5001, debug=True)