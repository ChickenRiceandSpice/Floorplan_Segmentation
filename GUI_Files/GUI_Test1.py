from nicegui import ui, app
from pathlib import Path

img_dir = Path('images')
img_dir.mkdir(exist_ok= True)

app.add_static_files('/images',img_dir)
