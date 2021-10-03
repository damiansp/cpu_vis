from tensorflow.keras.layers import (
    Dense, Flatten, GlobalAveragePooling2D, UpSampling2D)


# extract lower-level portion of resnet for retraining
def get_featured_extractors(inputs):
    feature_extractor_layer = tf.keras.applications.resnet.ResNet50(
        input_shape=(224, 224, 3), include_top=False, weights='imagenet'
    )(inputs)
    return feature_extractor_layer


def classifier(inputs):
    x = GlobalAveragePooling2D()(inputs)
    x = Flattern()(x)
    x = Dense(1024, activation='relu')(x)
    x = Dense(512, activation='relu')(x)
    x = Dense(10, activation='softmax' name='classification')(x)
    return x


def final_mod(inputs):
    # orig cifar (32 x 32) x (7 x 7) = imagenet (224 x 224)
    resize = UpSampling2D(size=(7, 7))(inputs) 
    resnet_feature_extractor = get_feature_extractor(resize)
    classification_outut = classifier(resnet_feature_extractor)
    return classificaton_output
