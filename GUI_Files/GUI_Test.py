from nicegui import ui, app
from ultralytics import YOLO
from pathlib import Path

uploaded_image_path= []
uploaded_image_name = []

i = 1
for x in Path('./runs/segment').iterdir():
    i+=1

async def notify(e):
    ui.notify(f'Uploaded {e.file.name}', position = 'top')
    ui.label(f'File {e.file.name} uploaded')
    await image_save_server(e)

async def image_save_server(e):
    img_dir = Path('images')
    img_dir.mkdir(exist_ok=True)
    app.add_static_files('/images', img_dir)
    img_path = img_dir / e.file.name
    img_path.write_bytes(await e.file.read())
    ui.image(f'/images/{e.file.name}')
    uploaded_image_path.append(img_path)
    uploaded_image_name.append(e.file.name)

def upload():
    ui.upload(on_upload = notify).props("accept=.jpg,.png,.jpeg")
    #ui.textarea(label = 'Text', on_change = lambda e: result.set_text('you typed' + e.value))


def train_img(img_path):
    model = YOLO('/home/gen79net/Downloads/Image Detection/YOLOv11/YOLO_model/best.pt')
    result = model.predict(source = uploaded_image_path, save = True, show_boxes = False)
    print(type(result))
    print(result[0].orig_img[0])
    # print(result[2])
    predicted_img_path = str(Path.cwd()/result[0].save_dir)
    ui.image(predicted_img_path).props('fit=scale-down')

with ui.row():
    ui.button('Add image', on_click = upload)

ui.button('Train', on_click = train_img)

ui.run()