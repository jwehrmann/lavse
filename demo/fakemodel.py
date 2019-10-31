from PIL import Image
from torchvision import transforms

#!flask/bin/python
from flask import Flask
from flask import Flask, request, redirect, url_for, flash, jsonify
from flask_cors import CORS, cross_origin

transform = transforms.Compose([
    transforms.Resize(256), transforms.CenterCrop(256)
])

def tob64(pil_img):
    from PIL import Image
    from io import BytesIO
    import base64
    # img = Image.fromarray(pil_img, 'RGB')
    buffer = BytesIO()
    pil_img.save(buffer,format="JPEG")
    myimage = buffer.getvalue()
    st = base64.b64encode(myimage).decode("utf-8")
    b64 = f"data:image/jpeg;base64,{st}"
    return b64


def fake_retrieve(query):

    img_list = ['0.png','1.png','2.png']

    images = [
        Image.open(img)
        for img
        in img_list
    ]
    imgs = []
    for i, image in enumerate(images):
        # image.save(f'{i}.png')
        image = transform(image)
        str_img = tob64(image)
        imgs.append(str_img)

    return jsonify({'response': imgs})


