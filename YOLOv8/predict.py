from ultralytics import YOLO
import PIL

# get image from datasets/images/test

image = PIL.Image.open("datasets/images/test/153.png")
model = YOLO("runs/detect/train/weights/best.pt")

# inference
result = model.predict(image)[0]
print(result.boxes)

res = result.plot(line_width=1, labels=True)
res = res[:,:,::-1]
res = PIL.Image.fromarray(res)
res.save("result.png")

