import os
from nicegui import ui
from ui.styles import *      # injects custom CSS (your dark theme)
from ui.pages import main_page   # registers the @ui.page('/') route

if __name__ in {"__main__", "__mp_main__"}:
    ui.run(
        host='0.0.0.0',
        port=int(os.environ.get('PORT', 8080)),
        title='Multi-Standard Engineering Auditor',
        favicon='🏗️',
        reload=False,          # set to True for local development
        reconnect_timeout=30.0,
    )
