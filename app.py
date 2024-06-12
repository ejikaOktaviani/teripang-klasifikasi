import numpy as np
import cv2
from PIL import Image
from rembg import remove
import joblib
import streamlit as st

# Implementasi Gaussian Naive Bayes
class GaussianNBManual:
    def fit(self, X, y):
        self.classes = np.unique(y)
        self.means = {}
        self.vars = {}
        self.priors = {}

        for c in self.classes:
            X_c = X[y == c]
            self.means[c] = np.mean(X_c, axis=0)
            self.vars[c] = np.var(X_c, axis=0)
            self.priors[c] = X_c.shape[0] / X.shape[0]

    def predict(self, X):
        y_pred = [self._predict(x) for x in X]
        return np.array(y_pred)

    def _predict(self, x):
        posteriors = []

        for c in self.classes:
            prior = np.log(self.priors[c])
            class_conditional = np.sum(np.log(self._pdf(c, x)))
            posterior = prior + class_conditional
            posteriors.append(posterior)

        return self.classes[np.argmax(posteriors)]

    def _pdf(self, class_idx, x):
        mean = self.means[class_idx]
        var = self.vars[class_idx]
        numerator = np.exp(- (x - mean) ** 2 / (2 * var))
        denominator = np.sqrt(2 * np.pi * var)
        return numerator / denominator

# Fungsi untuk resize gambar
def resize_image(input_image, size=(225, 225)):
    resized_img = input_image.resize(size)
    return resized_img

# Fungsi untuk menghapus latar belakang
def remove_background(input_image):
    output_image = remove(np.array(input_image))
    return Image.fromarray(output_image)

# Fungsi untuk menambahkan latar belakang putih
def add_white_background(image):
    image = image.convert("RGBA")
    background = Image.new("RGBA", image.size, (255, 255, 255, 255))
    combined = Image.alpha_composite(background, image)
    combined = combined.convert("RGB")  # Konversi ke RGB untuk menghilangkan alpha channel
    return combined

# Fungsi untuk konversi RGB ke HSV
def rgb_to_hsv(r, g, b):
    r, g, b = r / 255.0, g / 255.0, b / 255.0
    mx = max(r, g, b)
    mn = min(r, g, b)
    df = mx - mn
    if mx == mn:
        h = 0
    elif mx == r:
        h = (60 * ((g - b) / df) + 360) % 360
    elif mx == g:
        h = (60 * ((b - r) / df) + 120) % 360
    elif mx == b:
        h = (60 * ((r - g) / df) + 240) % 360
    if mx == 0:
        s = 0
    else:
        s = df / mx
    v = mx
    return h, s, v

# Fungsi untuk ekstraksi fitur warna dari gambar
def extract_color_features(image):
    image = image.convert('RGB')
    pixels = list(image.getdata())
    hsv_values = [rgb_to_hsv(r, g, b) for r, g, b in pixels]
    hsv_array = np.array(hsv_values)
    hsv_mean = np.mean(hsv_array, axis=0)
    return hsv_mean

# Fungsi untuk melakukan grayscale pada gambar
def grayscale(image):
    grayscale_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY)
    return grayscale_image

# Fungsi untuk ekstraksi fitur GLCM dari gambar
def ekstraksi_glcm(image):
    gray_image = grayscale(image)
    max_value = gray_image.max() + 1
    glcm = np.zeros((max_value, max_value))

    for i in range(gray_image.shape[0] - 1):
        for j in range(gray_image.shape[1] - 1):
            glcm[gray_image[i, j], gray_image[i + 1, j + 1]] += 1

    glcm /= np.sum(glcm)

    energy = np.sum(glcm**2)
    correlation = np.sum(glcm * np.arange(0, glcm.shape[0])[:, None])
    contrast = np.sum(glcm * (np.arange(0, glcm.shape[0])[:, None] - np.arange(0, glcm.shape[1]))**2)
    homogeneity = np.sum(glcm / (1 + np.abs(np.arange(0, glcm.shape[0])[:, None] - np.arange(0, glcm.shape[1]))))

    return np.array([energy, correlation, contrast, homogeneity])

# Fungsi utama untuk memproses gambar dan melakukan prediksi
def predict_teripang(image, model_path='NB.joblib'):
    # Step 1: Resize Image
    resized_image = resize_image(image)

    # Step 2: Remove Background
    no_bg_image = remove_background(resized_image)

    # Step 3: Add White Background
    white_bg_image = add_white_background(no_bg_image)

    # Step 4: Extract Color Features
    hsv_mean = extract_color_features(white_bg_image)

    # Step 5: Extract GLCM Features
    glcm_features = ekstraksi_glcm(white_bg_image)

    # Combine all features
    features = np.hstack((hsv_mean, glcm_features))

    # Load Model and Predict
    model = joblib.load(model_path)
    prediction = model.predict([features])

    return prediction

# Streamlit app
st.set_page_config(page_title="Teripang Classification App", page_icon="sea-cucumber.png", layout="wide")

# Sidebar
st.sidebar.title("Teripang Classification")
uploaded_file = st.sidebar.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])

# Main content
st.title("Welcome to Teripang Classifier!")
st.write("This app can classify different types of sea cucumbers (teripang).")

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    prediction = predict_teripang(image)

    st.subheader("Prediction:")
    st.write(f'The uploaded image is classified as: **{prediction[0]}**')

# Display model information
st.sidebar.title("Model Information")
st.sidebar.write("Trained using a dataset of sea cucumber images.")
st.sidebar.write("Algorithm: Gaussian Naive Bayes")
st.sidebar.write("Features used: Color Features (HSV), GLCM (Energy, Correlation, Contrast, Homogeneity)")
