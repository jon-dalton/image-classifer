import cv2
import tensorflow as tf

CATEGORIES = ["clear","dirt","snow","wet"]
            
def prepare(file):
    IMG_SIZE = 100
    img_array = cv2.imread(file, cv2.IMREAD_GRAYSCALE)
    new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
    return new_array.reshape(-1, IMG_SIZE, IMG_SIZE, 1)

model = tf.keras.models.load_model("CNN.model")
image = "test2.jpg" #your image path
new_image = prepare(image)
prediction = model.predict([new_image])
prediction = list(prediction[0])
print(CATEGORIES[prediction.index(max(prediction))])