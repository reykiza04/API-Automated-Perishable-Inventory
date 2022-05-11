import json
import tensorflow as tf
from flask import Flask, request, Response


app = Flask(__name__)

def load_meat_model():
    return tf.keras.models.load_model('models/model_meat.h5')

model_meat = load_meat_model()


@app.route('/inference', methods=['GET', 'POST'])
def meat():
    if request.method == 'POST':
        file = request.files['file']
        img = load_image(file.read(), size=(170, 170))
        clazz, prob = parse_meat_output(model_meat(img).numpy()[0])
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

def parse_meat_output(output):
    config ={
        0 : 'Fresh Meat Products',
        1 : 'Spoiled Meat Products',
        }
    
    clazz = output[0] > .5
    if clazz == 0:
        return config[0], 100 - int(output[0] * 100)
    else:
        return config[1], int(output[0] * 100)


# app.run(port=5000, debug=True)