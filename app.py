# app.py
import streamlit as st
import cv2
import numpy as np
import os
import random
import tempfile
from PIL import Image

# Streamlit page setup
st.set_page_config(page_title="Gum Image Augmentation", layout="wide")
st.title("🦷 Gum Image Augmentation App")
st.write("Upload or capture intraoral images to segment gum regions and generate augmented versions.")

# --- Sidebar ---
st.sidebar.header("⚙️ Augmentation Settings")
num_aug = st.sidebar.slider("Number of augmentations per image", 1, 100, 10)
kernel_size = st.sidebar.slider("Morphological kernel size", 3, 15, 7)
blur_values = st.sidebar.multiselect("Blur kernel options", [3, 5, 7], default=[3, 5, 7])
augment_types = st.sidebar.multiselect(
    "Augmentation Types",
    ["flip", "rotate", "brightness", "contrast", "noise", "blur"],
    default=["flip", "rotate", "brightness", "contrast", "noise", "blur"]
)

# --- Image Input Options ---
st.sidebar.header("📸 Image Source")
input_mode = st.sidebar.radio("Select input method:", ["Upload Images", "Use Camera"])

uploaded_files = None
captured_image = None

if input_mode == "Upload Images":
    uploaded_files = st.file_uploader("📂 Upload image(s)", type=["jpg", "jpeg", "png"], accept_multiple_files=True)
elif input_mode == "Use Camera":
    captured_image = st.camera_input("Take a photo with your webcam")

# --- Function to process and augment an image ---
def process_and_augment(img, file_name="captured_image.png"):
    output_dir = tempfile.mkdtemp()

    # Step 1: Segment gum region (HSV red range)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lower_red1, upper_red1 = np.array([0, 50, 50]), np.array([10, 255, 255])
    lower_red2, upper_red2 = np.array([160, 50, 50]), np.array([179, 255, 255])
    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    mask = cv2.bitwise_or(mask1, mask2)

    # Step 2: Morphological cleanup
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask = cv2.medianBlur(mask, 5)

    # Step 3: Apply mask to keep only gums
    gum_only = cv2.bitwise_and(img, img, mask=mask)

    # Display original and segmented image
    col1, col2 = st.columns(2)
    with col1:
        st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), caption="Original Image")
    with col2:
        st.image(cv2.cvtColor(gum_only, cv2.COLOR_BGR2RGB), caption="Segmented Gum Region")

    # Step 4: Generate augmentations
    st.write(f"Generating {num_aug} augmentations for **{file_name}**...")
    for i in range(num_aug):
        aug = gum_only.copy()
        aug_type = random.choice(augment_types)

        if aug_type == "flip":
            flip_code = random.choice([-1, 0, 1])
            aug = cv2.flip(aug, flip_code)

        elif aug_type == "rotate":
            h, w = aug.shape[:2]
            angle = random.randint(-20, 20)
            M = cv2.getRotationMatrix2D((w//2, h//2), angle, 1)
            aug = cv2.warpAffine(aug, M, (w, h))

        elif aug_type == "brightness":
            value = random.randint(-40, 40)
            hsv_aug = cv2.cvtColor(aug, cv2.COLOR_BGR2HSV)
            hsv_aug = hsv_aug.astype(np.int16)
            hsv_aug[:, :, 2] = np.clip(hsv_aug[:, :, 2] + value, 0, 255)
            hsv_aug = hsv_aug.astype(np.uint8)
            aug = cv2.cvtColor(hsv_aug, cv2.COLOR_HSV2BGR)

        elif aug_type == "contrast":
            alpha = random.uniform(0.6, 1.6)
            aug = np.clip(alpha * aug, 0, 255).astype(np.uint8)

        elif aug_type == "noise":
            noise = np.random.normal(0, 10, aug.shape).astype(np.uint8)
            aug = cv2.add(aug, noise)

        elif aug_type == "blur":
            k = random.choice(blur_values)
            aug = cv2.GaussianBlur(aug, (k, k), 0)

        save_name = f"{os.path.splitext(file_name)[0]}_aug_{i+1:03d}_{aug_type}.png"
        save_path = os.path.join(output_dir, save_name)
        cv2.imwrite(save_path, aug)

        if i < 3:
            st.image(cv2.cvtColor(aug, cv2.COLOR_BGR2RGB), caption=f"Aug #{i+1}: {aug_type}")

    st.success(f"✅ Finished augmenting {file_name}")
    st.info(f"🎯 Saved augmented images in: `{output_dir}`")

# --- Run processing ---
if st.button("🚀 Start Augmentation"):
    if input_mode == "Upload Images" and uploaded_files:
        for uploaded_file in uploaded_files:
            file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
            img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
            if img is not None:
                process_and_augment(img, uploaded_file.name)
    elif input_mode == "Use Camera" and captured_image:
        img = np.array(Image.open(captured_image))
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        process_and_augment(img, "captured_image.png")
    else:
        st.warning("⚠️ Please upload an image or take a photo first.")

