import streamlit as st
from glob import glob
import torch
import easyocr
import numpy as np
import cv2
from PIL import ImageFont, ImageDraw, Image
from tempfile import NamedTemporaryFile

st.set_page_config(layout='wide')


def main():
    car_m, lp_m, reader = load_model()
    st.title("불법 주정차 차량 번호판 인식")
    st.write("파일명 형식 ex) 2023.05.05.19;40;35.jpg")
    car_img = st.file_uploader(label='이미지 업로드', accept_multiple_files=True)

    for file in car_img:
        im, text = detect(car_m, lp_m, reader, file)
        if not text:  # 번호판 탐지가 안됐을 경우
            continue
        if len(sorted(text, reverse=True)[0]) < 7 or len(text) > 1:  # 번호탄 인식 결과가 여러개일 경우 또는 인식 문자가 7개 미만일 경우
            continue
        try:
            int(sorted(text[0], reverse=True)[0])
            continue
        except:
            pass
        with NamedTemporaryFile(dir='.', suffix='.jpg') as f:
            f.write(file.getbuffer())
        time = str(file).split("name='")[1]
        time = time.split(".jpg")[0]
        file_name = f.name.split('tmp')[0] + time + '.jpg'
        file_name = file_name.replace('\\', '/')
        st.image(im)
        lp = st.text_input("수정값을 입력하세요.", key=time + 'b')
        if lp:
            st.write('번호판 인식 결과 : ', lp)

            with open(f.name, 'wb') as f2:
                st.download_button(label="파일 다운로드", data=file, file_name=file.name.split('.jpg')[0] + ' ' + lp + '.jpg',
                                   mime="image/jpg")
        else:
            st.write('번호판 인식 결과 : ', *text)

            with open(f.name, 'wb') as f2:
                st.download_button(label="파일 다운로드", data=file,
                                   file_name=file.name.split('.jpg')[0] + ' ' + ''.join(str(s) for s in text) + '.jpg',
                                   mime="image/jpg")

        st.write(f'촬영 시각 : {time[0:4]}년 {time[5:7]}월 {time[8:10]}일 {time[11:13]}시 {time[14:16]}분 {time[17:19]}초')

        st.write('-------------------------------------')


@st.cache
def load_model():
    car_m = torch.hub.load("ultralytics/yolov5", 'yolov5s', force_reload=True, skip_validation=True)
    lp_m = torch.hub.load('ultralytics/yolov5', 'custom', 'lp_det.pt')
    reader = easyocr.Reader(['en'], detect_network='craft', recog_network='best_acc',
                            user_network_directory='lp_models/user_network', model_storage_directory='lp_models/models')

    car_m.classes = [2, 3, 5, 7]
    return car_m, lp_m, reader


import os


def detect(car_m, lp_m, reader, path):
    fontpath = "SpoqaHanSansNeo-Light.ttf"
    font = ImageFont.truetype(fontpath, 35)
    im = Image.open(path)

    im.save(path)
    to_draw = np.array(im)
    results = car_m(im)
    locs = results.xyxy[0]

    result_text = []

    if len(locs) == 0:

        result = lp_m(im)
        if len(result) == 0:
            result_text.append('검출된 차 없음')
        elif len(result) == 1:
            for rslt in result.xyxy[0]:
                x2, y2, x3, y3 = [item1.cpu().detach().numpy().astype(np.int32) for item1 in rslt[:4]]
                try:
                    extra_boxes = 0
                    im = cv2.cvtColor(
                        cv2.resize(to_draw[y2 - extra_boxes:y3 + extra_boxes, x2 - extra_boxes:x3 + extra_boxes],
                                   (224, 128)), cv2.COLOR_BGR2GRAY)
                    text = reader.recognize(im)[0][1]
                    result_text.append(text)
                except Exception as e:
                    return cv2.resize(to_draw, (480, 360)), ""
                img_pil = Image.fromarray(to_draw)
                draw = ImageDraw.Draw(img_pil)
                if y2 < 295:
                    draw.text((x2 - 40, y2 + 40), text, font=font, fill=(255, 255, 255))
                else:
                    draw.text((x2 - 40, y2 - 40), text, font=font, fill=(255, 255, 255))
                to_draw = np.array(img_pil)
                st.write((x2.item(), y2))
                cv2.rectangle(to_draw, (x2.item(), y2.item()), (x3.item(), y3.item()), (255, 255, 255), thickness=2)

            return cv2.resize(to_draw, (480, 360)), result_text

    for idx, item in enumerate(locs):
        x, y, x1, y1 = [it.cpu().detach().numpy().astype(np.int32) for it in item[:4]]
        car_im = to_draw[y:y1, x:x1, :].copy()
        result = lp_m(Image.fromarray(car_im))

        if len(result) == 0:
            result_text.append("차는 검출됬으나, 번호판이 검출되지 않음")

        for rslt in result.xyxy[0]:
            x2, y2, x3, y3 = [item1.cpu().detach().numpy().astype(np.int32) for item1 in rslt[:4]]
            try:
                extra_boxes = 0
                im = cv2.cvtColor(cv2.resize(
                    to_draw[y + y2 - extra_boxes:y + y3 + extra_boxes, x + x2 - extra_boxes:x + x3 + extra_boxes],
                    (224, 128)), cv2.COLOR_BGR2GRAY)
                text = reader.recognize(im)[0][1]
                result_text.append(text)
            except Exception as e:
                return cv2.resize(to_draw, (480, 360)), ""
            img_pil = Image.fromarray(to_draw)
            draw = ImageDraw.Draw(img_pil)
            if y+y2 < 295:
                draw.text((x + x2 - 40, y + y2 + 40), text, font=font, fill=(255, 255, 255))
            else:
                draw.text((x + x2 - 40, y + y2 - 40), text, font=font, fill=(255, 255, 255))
            to_draw = np.array(img_pil)
            cv2.rectangle(to_draw, (x + x2, y + y2), (x + x3, y + y3), (255, 255, 255), thickness=2)

    return cv2.resize(to_draw, (480, 360)), result_text


if __name__ == '__main__':
    main()
