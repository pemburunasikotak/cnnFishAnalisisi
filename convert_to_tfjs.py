import sys
from types import ModuleType

# Mock missing heavy dependencies not needed for Keras H5 conversion
sys.modules['tensorflow_decision_forests'] = ModuleType('tensorflow_decision_forests')
sys.modules['yggdrasil_decision_forests'] = ModuleType('yggdrasil_decision_forests')
sys.modules['yggdrasil_decision_forests.dataset'] = ModuleType('yggdrasil_decision_forests.dataset')
sys.modules['jax'] = ModuleType('jax')

jax_experimental = ModuleType('jax.experimental')
jax_experimental.jax2tf = lambda *args, **kwargs: None
sys.modules['jax.experimental'] = jax_experimental

import tensorflow as tf
import tensorflowjs as tfjs

print("Loading Keras model 'model_prediksi_ikan.h5'...")
try:
    model = tf.keras.models.load_model('model_prediksi_ikan.h5')
    print("Model loaded successfully!")
except Exception as e:
    print(f"Error loading model: {e}")
    sys.exit(1)

print("Converting and saving model to 'model_tfjs' folder...")
try:
    tfjs.converters.save_keras_model(model, 'model_tfjs')
    print("Conversion successful! Files saved under 'model_tfjs/'")
except Exception as e:
    print(f"Error during conversion: {e}")
    sys.exit(1)
