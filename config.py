import os
from dotenv import load_dotenv
from google import genai
from google.genai import types
from reportlab.lib.pagesizes import A4
import re   # if not already imported

_LATEX_SIMPLE = {
    r'\times': ' x ', r'\cdot': ' . ', r'\div': ' / ',
    r'\geq': ' >= ', r'\ge': ' >= ', r'\leq': ' <= ', r'\le': ' <= ',
    r'\pm': ' +/- ', r'\approx': ' ~= ', r'\neq': ' != ',
    r'\infty': 'infinity', r'\text': '', r'\mathrm': '', r'\mathbf': '',
    r'\left': '', r'\right': '', r'\,': ' ', r'\;': ' ', r'\!': '',
    r'\Delta': 'Delta ', r'\delta': 'delta ', r'\sigma': 'sigma ', r'\Sigma': 'Sigma ',
    r'\phi': 'phi ', r'\gamma': 'gamma ', r'\theta': 'theta ', r'\mu': 'mu ',
    r'\pi': 'pi ', r'\alpha': 'alpha ', r'\beta': 'beta ', r'\rho': 'rho ',
    r'\max': 'Max', r'\min': 'Min', r'\sum': 'Sum', r'\bar': '',
}

def sanitize_ai_markdown(text: str) -> str:
    if not text:
        return ""
    text = str(text)
    for macro, repl in _LATEX_SIMPLE.items():
        text = text.replace(macro, repl)
    for _ in range(2):
        text = re.sub(r'\\frac\s*\{([^{}]*)\}\s*\{([^{}]*)\}', r'(\1 / \2)', text)
        text = re.sub(r'\\sqrt\s*\{([^{}]*)\}', r'sqrt(\1)', text)
    text = re.sub(r'_\{([^{}]*)\}', r'_\1', text)
    text = re.sub(r'\^\{([^{}]*)\}', r'^\1', text)
    text = re.sub(r'\\([a-zA-Z]+)', r'\1', text)
    text = text.replace('$$', '').replace('$', '')
    text = re.sub(r'(?<!\w)\{([^{}]{0,40})\}(?!\w)', r'\1', text)
    text = re.sub(r'\*{3,}', '**', text)
    text = re.sub(r'([^\n])\n(#{1,6}\s)', r'\1\n\n\2', text)
    text = re.sub(r'([^\n|])\n(\|)', r'\1\n\n\2', text)
    text = re.sub(r'(?<![\w#*`|])[&$%^~?/\\]{2,}(?![\w#*`|])', '', text)
    text = re.sub(r'[ \t]+\n', '\n', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()

load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")
client = genai.Client(api_key=api_key) if api_key else None
GEMINI_MODEL = "gemini-3.5-flash-lite"

# ReportLab page constants
PAGE_WIDTH, PAGE_HEIGHT = A4
MARGIN = 32
USABLE_WIDTH = PAGE_WIDTH - (2 * MARGIN)

# Code basis options
CODE_BASIS_OPTIONS = [
    "Egyptian Codes: ECP 203 / ECP 202 / ECP 104 (Default Core Basis)",
    "ACI 318-25 — Structural Concrete (Primary)",
    "Eurocode 2 — BS EN 1992 + UK Annex (Primary)",
    "AASHTO LRFD Bridge & Pavement Design (Primary)",
    "IBC — International Building Code (Primary)",
]

def get_code_directive(basis: str) -> str:
    if not basis or basis.startswith("Egyptian Codes"):
        return (
            "GOVERNING STANDARD (MANDATORY): Base every clause reference, formula, allowable limit and "
            "pass/fail verdict strictly on the Egyptian Codes — ECP 203 (Reinforced Concrete Structures), "
            "ECP 202 (Soil Mechanics & Foundations) and ECP 104 (Roads, Highways & Airfields), as applicable "
            "to the topic. Do NOT substitute ACI, Eurocode or AASHTO limits. Where relevant, cite the specific "
            "ECP clause, table or article number."
        )
    return (
        f"GOVERNING STANDARD (MANDATORY): The user has selected an alternative primary design basis: "
        f"\"{basis}\". Use that standard as the PRIMARY source for formulas, limits, and clause citations. "
        "Mention the equivalent Egyptian Code (ECP 203 / 202 / 104) clause only as a secondary cross-reference."
    )

NO_LATEX_RULE = (
    "OUTPUT FORMAT (MANDATORY): Write in clean GitHub-flavoured Markdown only. "
    "Never use LaTeX, dollar-sign math delimiters ($ or $$), backslash commands (\\frac, \\times, \\ge ...), "
    "or curly-brace variable syntax. Write formulas in plain readable text, e.g. 'f_cu = 30 N/mm2', "
    "'Standard Deviation = 2.1 N/mm2'. Use real Markdown tables (with a header row and a --- separator row) "
    "for any tabular data — never hand-draw tables with dashes or asterisks. Use ## / ### for section headings, "
    "never #### or deeper. Use single asterisks pairs (**bold**) and never stack more than two."
)

_LATEX_SIMPLE = {
    r'\times': ' x ', r'\cdot': ' . ', r'\div': ' / ',
    r'\geq': ' >= ', r'\ge': ' >= ', r'\leq': ' <= ', r'\le': ' <= ',
    r'\pm': ' +/- ', r'\approx': ' ~= ', r'\neq': ' != ',
    r'\infty': 'infinity', r'\text': '', r'\mathrm': '', r'\mathbf': '',
    r'\left': '', r'\right': '', r'\,': ' ', r'\;': ' ', r'\!': '',
    r'\Delta': 'Delta ', r'\delta': 'delta ', r'\sigma': 'sigma ', r'\Sigma': 'Sigma ',
    r'\phi': 'phi ', r'\gamma': 'gamma ', r'\theta': 'theta ', r'\mu': 'mu ',
    r'\pi': 'pi ', r'\alpha': 'alpha ', r'\beta': 'beta ', r'\rho': 'rho ',
    r'\max': 'Max', r'\min': 'Min', r'\sum': 'Sum', r'\bar': '',
}

# BOQ Unit Rates (Architectural & Structural)
UNIT_RATES = {
    "Flooring (Ceramic)": 150,
    "Flooring (Marble)": 500,
    "Flooring (Tiles)": 200,
    "Wall Finishing (Paint)": 30,
    "Wall Finishing (Plaster)": 80,
    "Ceiling (Paint)": 25,
    "Ceiling (Gypsum Board)": 120,
    "Skirting (Ceramic)": 60,
    "Skirting (Marble)": 200,
    "Doors (Wood)": 3000,
    "Windows (Aluminum)": 2000,
    "Partitions (Gypsum)": 150,
    "Concrete (C30/37)": 2500,
    "Concrete (C25/30)": 2200,
    "Concrete (C40/50)": 3000,
    "Rebar (Grade 400)": 15000,
    "Rebar (Grade 600)": 18000,
    "Formwork": 300,
    "Excavation": 200,
    "Backfill": 150,
    "Foundation Concrete": 2800,
}

# For enhanced AutoCAD generator
BOQ_RATES = {
    'concrete_m3': 2500.0,
    'rebar_ton': 15000.0,
    'formwork_m2': 300.0,
    'bricks_nos': 2.5,
    'flooring_m2': 150.0,
    'paint_m2': 30.0,
    'plaster_m2': 80.0,
}

FIELD_LABELS = {
    'width_mm': 'Width (mm)',
    'depth_mm': 'Depth (mm)',
    'height_mm': 'Height (mm)',
    'length_mm': 'Length (mm)',
    'thickness_mm': 'Thickness (mm)',
    'area_m2': 'Area (m²)',
    'length_m': 'Length (m)',
    'height_m': 'Height (m)',
    'count': 'Count (number of columns/beams)',
    'main_diameter_mm': 'Main Bar Diameter (mm)',
    'stirrup_diameter_mm': 'Stirrup Diameter (mm)',
    'spacing_mm': 'Spacing (mm)',
    'top_diameter_mm': 'Top Bar Diameter (mm)',
    'bottom_diameter_mm': 'Bottom Bar Diameter (mm)',
}

MASS_SCHEMAS = {
    'columns': {
        'required': ['label', 'count', 'width_mm', 'depth_mm', 'height_mm'],
        'field_aliases': {
            'width': 'width_mm',
            'depth': 'depth_mm',
            'height': 'height_mm',
            'width_mm': 'width_mm',
            'depth_mm': 'depth_mm',
            'height_mm': 'height_mm',
            'count': 'count',
        }
    },
    'beams': {
        'required': ['label', 'count', 'width_mm', 'depth_mm', 'length_mm'],
        'field_aliases': {
            'width': 'width_mm',
            'depth': 'depth_mm',
            'length': 'length_mm',
            'width_mm': 'width_mm',
            'depth_mm': 'depth_mm',
            'length_mm': 'length_mm',
            'count': 'count',
        }
    },
    'slabs': {
        'required': ['label', 'thickness_mm', 'area_m2'],
        'field_aliases': {
            'thickness': 'thickness_mm',
            'area': 'area_m2',
            'thickness_mm': 'thickness_mm',
            'area_m2': 'area_m2',
        }
    },
    'footings': {
        'required': ['label', 'count', 'width_mm', 'depth_mm', 'length_mm'],
        'field_aliases': {
            'width': 'width_mm',
            'depth': 'depth_mm',
            'length': 'length_mm',
            'width_mm': 'width_mm',
            'depth_mm': 'depth_mm',
            'length_mm': 'length_mm',
            'count': 'count',
        }
    },
    'walls': {
        'required': ['label', 'count', 'length_m', 'height_m', 'thickness_mm'],
        'field_aliases': {
            'length': 'length_m',
            'height': 'height_m',
            'thickness': 'thickness_mm',
            'length_m': 'length_m',
            'height_m': 'height_m',
            'thickness_mm': 'thickness_mm',
            'count': 'count',
        }
    }
}

ARCH_SCHEMAS = {
    'flooring': {
        'required': ['total_length_m', 'total_width_m', 'area_m2'],
        'field_aliases': {
            'length': 'total_length_m',
            'width': 'total_width_m',
            'area': 'area_m2',
            'total_length_m': 'total_length_m',
            'total_width_m': 'total_width_m',
            'area_m2': 'area_m2',
        },
        'formula': lambda data: data.get('area_m2') if data.get('area_m2') else (data.get('total_length_m', 0) * data.get('total_width_m', 0))
    },
    'wall_finishing': {
        'required': ['total_area_m2'],
        'field_aliases': {
            'area': 'total_area_m2',
            'total_area_m2': 'total_area_m2',
        },
        'formula': lambda data: data.get('total_area_m2', 0)
    },
    'ceilings': {
        'required': ['total_area_m2'],
        'field_aliases': {
            'area': 'total_area_m2',
            'total_area_m2': 'total_area_m2',
        },
        'formula': lambda data: data.get('total_area_m2', 0)
    },
    'doors_windows': {
        'required': ['door_count', 'window_count'],
        'field_aliases': {
            'doors': 'door_count',
            'windows': 'window_count',
            'door_count': 'door_count',
            'window_count': 'window_count',
        },
        'formula': lambda data: (data.get('door_count', 0), data.get('window_count', 0))
    }
}
