from keras.models import load_model
from keras.utils import load_img, img_to_array
from tensorflow import expand_dims
from PIL import Image, ImageDraw, ImageFont

loaded_model = load_model("modeled.keras")

test_images = [
    "Data/Test/Medium/Crack__20180419_06_19_09,915.bmp",
    "Data/Test/Large/Crack__20180419_13_29_14,846.bmp"
]

labels = ["Large Crack", "Medium Crack", "Small Crack", "No Crack"]

for image_path in test_images:
    input_image = Image.open(image_path).convert("RGB")

    test_img = load_img(image_path, target_size=(100, 100))
    img_array = img_to_array(test_img)
    img_array = expand_dims(img_array, 0)
    img_array = img_array / 255.

    predictions = loaded_model.predict(img_array)

    for label, prob in zip(labels, predictions[0]):
        print(f"{label}: {prob * 100:.2f}%")

    draw_image = ImageDraw.Draw(input_image)
    for i, label in enumerate(labels):
        draw_image.text((1000, 1500 + i * 75), f"{label}: {round(predictions[0][i] * 100)}%", font = ImageFont.truetype("Arial.ttf", size=100), fill='white')

    filename = image_path.split('/')[-1]
    input_image.save("Prediction_"+filename.replace(".bmp","")+".jpeg", quality=75)
