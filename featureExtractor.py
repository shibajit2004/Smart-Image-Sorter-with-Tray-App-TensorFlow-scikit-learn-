import tensorflow as tf

feature_extractor = tf.keras.applications.MobileNetV2(
    include_top=False, weights='imagenet', input_shape=(128, 128, 3), pooling='avg'
)

feature_extractor.save("mobilenetv2_feature_extractor.keras")
print("Feature extractor saved.")