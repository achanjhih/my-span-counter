# app.py
import streamlit as st
import cv2
import numpy as np
from pdf2image import convert_from_bytes
from PIL import Image

st.title("PDF図面からスパン数を自動カウント")

uploaded_file = st.file_uploader("PDFファイルをアップロード", type=["pdf"])

if uploaded_file is not None:
    # PDFから画像に変換（1ページ目のみ）
    images = convert_from_bytes(uploaded_file.read(), first_page=1, last_page=1)
    img_pil = images[0]
    img = np.array(img_pil)

    # OpenCV形式へ変換
    img_cv = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    span_count = 0
    for cnt in contours:
        approx = cv2.approxPolyDP(cnt, 0.02 * cv2.arcLength(cnt, True), True)
        if len(approx) == 4 and cv2.contourArea(cnt) > 1000:
            span_count += 1
            cv2.drawContours(img_cv, [approx], 0, (0, 0, 255), 3)

    # 結果表示
    st.subheader(f"検出されたスパン数：{span_count}")
    result_img = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)
    st.image(result_img, caption="検出結果", use_column_width=True)
