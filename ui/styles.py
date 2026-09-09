from nicegui import ui

ui.add_head_html('''
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
    ::-webkit-scrollbar { width: 8px !important; background: #031338 !important; }
    ::-webkit-scrollbar-thumb { background: #FF8C00 !important; border-radius: 10px; }
    html, body {
        background: radial-gradient(circle at 10% 20%, #0a1a3a, #031338) !important;
        color: #E9EDF5 !important;
        font-family: 'Inter', 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif !important;
        margin: 0; padding: 0;
        width: 100vw; height: 100vh;
        overflow-x: hidden;
    }
    .sidebar-container {
        background: #0b1a3a !important;
        border-right: 2px solid rgba(255, 140, 0, 0.4) !important;
        box-shadow: 8px 0 30px rgba(0,0,0,0.6) !important;
    }
    .sidebar-container .q-field__control {
        background-color: rgba(13, 26, 53, 0.8) !important;
        border: 1px solid #2c3f6b !important;
        border-radius: 10px !important;
    }
    .sidebar-container .q-field__native,
    .sidebar-container .q-field__input,
    .sidebar-container .q-field__label {
        color: #E9EDF5 !important;
    }
    .sidebar-container .q-select .q-field__control {
        background-color: rgba(13, 26, 53, 0.8) !important;
    }
    .output-card {
        background: transparent !important;
        border: none !important;
        padding: 0 !important;
        box-shadow: none !important;
        width: 100% !important;
        max-width: none !important;
        box-sizing: border-box;
    }
    .input-card {
        background: rgba(13, 26, 53, 0.6);
        backdrop-filter: blur(8px);
        border: 1px solid rgba(255, 140, 0, 0.2);
        border-radius: 16px;
        padding: 18px 22px;
        box-shadow: 0 8px 32px rgba(0,0,0,0.3);
        margin-bottom: 20px;
        width: 100% !important;
        max-width: none !important;
        box-sizing: border-box;
    }
    .primary-btn, .q-btn {
        background: linear-gradient(135deg, #1a1a1a 0%, #333333 100%) !important;
        color: #FFFFFF !important;
        border: 1px solid #555 !important;
        font-weight: 600 !important;
        border-radius: 14px !important;
        padding: 10px 28px !important;
        letter-spacing: .4px;
        text-transform: none !important;
        box-shadow: 0 4px 15px rgba(0,0,0,0.5) !important;
        transition: all 0.25s ease !important;
        min-height: 44px !important;
    }
    .primary-btn:hover, .q-btn:hover {
        background: linear-gradient(135deg, #2d2d2d 0%, #444444 100%) !important;
        transform: translateY(-3px) !important;
        box-shadow: 0 8px 25px rgba(0,0,0,0.7) !important;
        border-color: #FF8C00 !important;
    }
    .primary-btn:active, .q-btn:active {
        transform: translateY(0px) !important;
    }
    .q-uploader {
        background: rgba(13, 26, 53, 0.6) !important;
        backdrop-filter: blur(8px) !important;
        border-radius: 14px !important;
        border: 2px dashed rgba(255, 140, 0, 0.5) !important;
        color: #FFFFFF !important;
        padding: 8px !important;
    }
    .q-uploader .q-uploader__header {
        background: transparent !important;
        color: #FFFFFF !important;
    }
    .q-uploader .q-uploader__header-content {
        color: #FFFFFF !important;
    }
    .q-uploader .q-uploader__file {
        background: rgba(13, 26, 53, 0.8) !important;
        color: #FFFFFF !important;
        border-radius: 10px !important;
    }
    input, select, textarea, .q-field__control {
        background-color: rgba(13, 26, 53, 0.7) !important;
        color: #FFFFFF !important;
        border: 1px solid #2c3f6b !important;
        border-radius: 10px !important;
    }
    .q-field__native, .q-field__input, .q-field__label {
        color: #E9EDF5 !important;
    }
    .q-field--highlighted .q-field__label {
        color: #FF8C00 !important;
    }
    .q-menu, .q-popover, .q-virtual-scroll__content {
        background: rgba(13, 26, 53, 0.95) !important;
        backdrop-filter: blur(8px) !important;
        border: 1px solid #2c3f6b !important;
        border-radius: 10px !important;
    }
    .q-item {
        color: #FFFFFF !important;
        background: transparent !important;
        border-radius: 8px !important;
    }
    .q-item:hover {
        background: rgba(255, 140, 0, 0.15) !important;
        color: #FF8C00 !important;
    }
    .app-footer {
        width: 100%;
        background: rgba(13, 26, 53, 0.7);
        backdrop-filter: blur(8px);
        border-top: 2px solid rgba(255, 140, 0, 0.5);
        padding: 20px 24px;
        margin-top: 50px;
        text-align: center;
        color: #A9B6D0;
        font-size: 13px;
        box-sizing: border-box;
        border-radius: 16px 16px 0 0;
    }
    .app-footer a {
        color: #4FC3F7;
        text-decoration: none;
        transition: color 0.3s ease;
    }
    .app-footer a:hover {
        color: #FF8C00;
        text-decoration: underline;
    }
    .markdown-body {
        font-size: 14px;
        line-height: 1.7;
        color: #E9EDF5;
        background: transparent !important;
        padding: 0 !important;
    }
    .markdown-body h1, .markdown-body h2, .markdown-body h3,
    .markdown-body h4, .markdown-body h5, .markdown-body h6 {
        font-family: 'Inter', sans-serif !important;
        font-weight: 700 !important;
        color: #FF8C00 !important;
        margin: 20px 0 10px 0 !important;
        line-height: 1.35 !important;
    }
    .markdown-body h1 { font-size: 22px !important; border-bottom: 2px solid #FF8C00; padding-bottom: 8px; }
    .markdown-body h2 { font-size: 19px !important; }
    .markdown-body h3 { font-size: 17px !important; color: #4FC3F7 !important; }
    .markdown-body h4, .markdown-body h5, .markdown-body h6 { font-size: 15px !important; color: #4FC3F7 !important; }
    .markdown-body p { margin: 10px 0 !important; }
    .markdown-body strong { color: #FFFFFF; }
    .markdown-body ul, .markdown-body ol { padding-left: 25px !important; margin: 10px 0 !important; }
    .markdown-body li { margin: 5px 0 !important; }
    .markdown-body hr { border-color: #1f3355; margin: 16px 0; }
    .markdown-body code {
        background: rgba(3, 19, 56, 0.8);
        border: 1px solid #1f3355;
        border-radius: 4px;
        padding: 2px 6px;
        font-size: 12.5px;
        color: #4FC3F7;
    }
    .markdown-body table {
        border-collapse: collapse !important;
        width: 100% !important;
        margin: 16px 0 !important;
        font-size: 13px !important;
        table-layout: auto !important;
        border-radius: 12px !important;
        overflow: hidden !important;
        box-shadow: 0 4px 15px rgba(0,0,0,0.3) !important;
    }
    .markdown-body th, .markdown-body td {
        border: 1px solid #1f3355 !important;
        padding: 10px 14px !important;
        text-align: left !important;
        word-wrap: break-word !important;
        white-space: normal !important;
    }
    .markdown-body th {
        background: linear-gradient(135deg, #1a1a1a 0%, #333333 100%) !important;
        color: #FF8C00 !important;
        font-weight: 700 !important;
    }
    .markdown-body tr:nth-child(even) td {
        background-color: rgba(10, 26, 58, 0.5);
    }
    .markdown-body tr:hover td {
        background-color: rgba(255, 140, 0, 0.08);
    }
    .stat-chip {
        background: rgba(13, 26, 53, 0.6);
        backdrop-filter: blur(8px);
        border: 1px solid #1f3355;
        border-radius: 12px;
        padding: 14px 20px;
        text-align: center;
        min-width: 140px;
        transition: all 0.3s ease;
    }
    .stat-chip:hover {
        border-color: #FF8C00;
        transform: translateY(-3px);
        box-shadow: 0 6px 20px rgba(255,140,0,0.15);
    }
    .stat-chip .val {
        font-size: 24px;
        font-weight: 800;
        color: #FF8C00;
    }
    .stat-chip .lbl {
        font-size: 11px;
        color: #A9B6D0;
        text-transform: uppercase;
        letter-spacing: .05em;
        margin-top: 4px;
    }
    .q-tabs {
        border-radius: 14px !important;
        overflow: hidden !important;
        background: rgba(13, 26, 53, 0.6) !important;
        backdrop-filter: blur(8px) !important;
        padding: 4px !important;
    }
    .q-tabs__content {
        overflow-x: auto !important;
        flex-wrap: nowrap !important;
        scrollbar-width: thin;
        scrollbar-color: #FF8C00 transparent;
    }
    .q-tabs__content::-webkit-scrollbar {
        height: 4px;
    }
    .q-tabs__content::-webkit-scrollbar-thumb {
        background: #FF8C00;
        border-radius: 2px;
    }
    .q-tabs__content::-webkit-scrollbar-track {
        background: transparent;
    }
    .q-tab {
        color: #A9B6D0 !important;
        font-weight: 600 !important;
        transition: all 0.3s ease !important;
        border-radius: 10px !important;
        margin: 2px !important;
        padding: 8px 16px !important;
        white-space: nowrap;
        flex-shrink: 0;
    }
    .q-tab:hover {
        color: #FFFFFF !important;
        background: rgba(255, 140, 0, 0.1) !important;
    }
    .q-tab--active {
        color: #FF8C00 !important;
        background: rgba(255, 140, 0, 0.15) !important;
    }
    .q-tab__indicator {
        background: #FF8C00 !important;
        height: 3px !important;
        border-radius: 2px !important;
    }
    .chat-message {
        padding: 8px 0;
        border-bottom: 1px solid rgba(255,255,255,0.05);
    }
    .chat-message:last-child {
        border-bottom: none;
    }
    .chat-message .role-label {
        font-size: 12px;
        font-weight: 700;
        margin-bottom: 2px;
    }
    .chat-message .role-label.assistant {
        color: #FF8C00;
    }
    .chat-message .role-label.user {
        color: #4FC3F7;
    }
    .chat-message .content {
        padding-left: 8px;
    }
    @media (max-width: 768px) {
        .markdown-body table {
            font-size: 11px !important;
        }
        .markdown-body th, .markdown-body td {
            padding: 6px 8px !important;
        }
        .stat-chip {
            min-width: 100px !important;
            padding: 10px 14px !important;
        }
        .stat-chip .val {
            font-size: 18px !important;
        }
        .input-card {
            padding: 12px 14px !important;
        }
        .primary-btn, .q-btn {
            padding: 8px 16px !important;
            font-size: 13px !important;
            min-height: 36px !important;
            border-radius: 10px !important;
        }
        .app-footer {
            font-size: 11px !important;
            padding: 14px 12px !important;
        }
        .q-tabs__content {
            flex-wrap: nowrap !important;
        }
        .q-tab {
            font-size: 12px !important;
            padding: 6px 10px !important;
        }
        .q-uploader {
            font-size: 12px !important;
        }
        .markdown-body {
            overflow-x: auto;
        }
        .markdown-body table {
            display: block;
            overflow-x: auto;
            white-space: nowrap;
        }
        .markdown-body table td, .markdown-body table th {
            white-space: normal !important;
        }
        .markdown-body {
            overflow-x: auto;
        }
    }
    .main-title {
        font-size: 3.8rem !important;
        font-weight: 900 !important;
        letter-spacing: -0.02em;
    }
    .sub-title {
        color: #FFFFFF !important;
        font-weight: 500;
    }
    @media (max-width: 768px) {
        .main-title {
            font-size: 2.2rem !important;
        }
        .sub-title {
            font-size: 1rem !important;
        }
    }
</style>
''', shared=True)
