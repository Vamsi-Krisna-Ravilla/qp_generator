import streamlit as st
import pandas as pd
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image as RLImage, Table, TableStyle, HRFlowable
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from io import BytesIO
import requests
from PIL import Image
import re
from enum import Enum
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
from datetime import datetime
import hmac
import pickle
from reportlab.lib.pdfencrypt import StandardEncryption

# UI Configuration
st.set_page_config(
    page_title="Question Paper Generator",
    page_icon="📝",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
        .stButton>button {
            width: 100%;
            border-radius: 5px;
            height: 3em;
        }
        .stSelectbox {
            border-radius: 5px;
        }
        .main-header {
            text-align: center;
            padding: 1rem;
            background-color: #f0f2f6;
            border-radius: 5px;
            margin-bottom: 1rem;
            background:black;
        }
        .section-header {
            background-color: #e7f3fe;
            padding: 0.5rem;
            border-radius: 5px;
            margin: 1rem 0;
            background:black;
        }
        .info-box {
            background-color: #e7f3fe;
            border-left: 6px solid #2196F3;
            padding: 1rem;
            margin: 1rem 0;
        }
    </style>
""", unsafe_allow_html=True)

# Constants
ACADEMIC_LEVELS = [
    "I B.Tech", "II B.Tech", "III B.Tech", "IV B.Tech",
    "I M.Tech", "II M.Tech", "I MCA", "II MCA","I BCA","II BCA","III BCA"
]

BRANCHES = [
    "CSE", "CSE(AIML)", "CSE(DS)", "ECE", "EVT"
]

MONTHS = [
    "JANUARY", "FEBRUARY", "MARCH", "APRIL", "MAY", "JUNE",
    "JULY", "AUGUST", "SEPTEMBER", "OCTOBER", "NOVEMBER", "DECEMBER"
]

class SubjectType(Enum):
    BEEE = "Basic Electrical & Electronics Engineering (RV23991T07)"
    BCME = "Basic Civil & Mechanical Engineering (RV23991T04)"
    EG = "Engineering Graphics (RV23991T08)"
    PHYSICS = "Engineering Physics (RV23991T06)"
    CHEMISTRY = "Engineering Chemistry (RV23991T02)"
    LINEAR_ALGEBRA_AND_CALCULUS = "Linear Algebra & Calculus (RV23991T03)"
    INTRODUCTION_TO_PROGRAMMING = "Introduction to Programming (RV23991T05)"
    ENGLISH = "Communicative English (RV23991T01)"

class ExamType(Enum):
    MID1_2 = "mid1 (Unit- 1, 2)"
    MID1_2_5 = "mid1 (Unit- 1, 2 & 3.1)"
    MID2_3_4_5 = "mid2 (Unit- 3, 4 & 5)"
    MID2_5_5 = "mid2 (Unit- 3.2, 4 & 5)"
    REGULAR = "regular"
    SUPPLY = "supply"

# Subject mapping based on academic level and semester
SEMESTER_SUBJECTS = {
    ("I B.Tech", "I"): [
        SubjectType.LINEAR_ALGEBRA_AND_CALCULUS,
        SubjectType.CHEMISTRY,
        SubjectType.BCME,
        SubjectType.INTRODUCTION_TO_PROGRAMMING,
        SubjectType.ENGLISH,
        SubjectType.PHYSICS,
        SubjectType.BEEE,
        SubjectType.EG
    ],
    ("I B.Tech", "II"): [
        SubjectType.LINEAR_ALGEBRA_AND_CALCULUS,
        SubjectType.CHEMISTRY,
        SubjectType.BCME,
        SubjectType.INTRODUCTION_TO_PROGRAMMING,
        SubjectType.ENGLISH,
        SubjectType.PHYSICS,
        SubjectType.BEEE,
        SubjectType.EG
    ]
}

@dataclass
class QuestionPattern:
    part_a: Dict[float, Dict[str, int]]
    part_b: Dict[float, Dict[str, int]]
    marks_a: Tuple[int, int]
    marks_b: Tuple[int, int]

# Define patterns
PATTERNS = {}

def add_general_subject_patterns(subject: SubjectType):
    """Add patterns for general subjects (not BCME/BEEE/EG)"""
    PATTERNS[(subject, ExamType.MID1_2)] = QuestionPattern(
        part_a={1: {'short': 3}, 2: {'short': 2}},
        part_b={1: {'long': 3}, 2: {'long': 3}},
        marks_a=(5, 2),
        marks_b=(3, 5)
    )
    
    PATTERNS[(subject, ExamType.MID1_2_5)] = QuestionPattern(
        part_a={1: {'short': 2}, 2: {'short': 2}, 3.1: {'short': 1}},
        part_b={1: {'long': 2}, 2: {'long': 2}, 3.1: {'long': 2}},
        marks_a=(5, 2),
        marks_b=(3, 5)
    )
    
    PATTERNS[(subject, ExamType.MID2_3_4_5)] = QuestionPattern(
        part_a={3: {'short': 2}, 4: {'short': 2}, 5: {'short': 1}},
        part_b={3: {'long': 2}, 4: {'long': 2}, 5: {'long': 2}},
        marks_a=(5, 2),
        marks_b=(3, 5)
    )
    
    PATTERNS[(subject, ExamType.MID2_5_5)] = QuestionPattern(
        part_a={3.2: {'short': 2}, 4: {'short': 2}, 5: {'short': 1}},
        part_b={3.2: {'long': 2}, 4: {'long': 2}, 5: {'long': 2}},
        marks_a=(5, 2),
        marks_b=(3, 5)
    )
    
    PATTERNS[(subject, ExamType.REGULAR)] = QuestionPattern(
        part_a={1: {'short': 2}, 2: {'short': 2}, 3.1: {'short': 1}, 
               3.2: {'short': 1}, 4: {'short': 2}, 5: {'short': 2}},
        part_b={1: {'long': 2}, 2: {'long': 2}, 3.1: {'long': 1}, 
               3.2: {'long': 1}, 4: {'long': 2}, 5: {'long': 2}},
        marks_a=(10, 2),
        marks_b=(5, 10)
    )

def add_bcme_beee_patterns(subject: SubjectType):
    """Add patterns for BCME/BEEE"""
    PATTERNS[(subject, ExamType.MID1_2_5)] = QuestionPattern(
        part_a={1: {'short': 2}, 2: {'short': 2}, 3: {'short': 1}},
        part_b={1: {'long': 2}, 2: {'long': 2}, 3: {'long': 2}},
        marks_a=(5, 2),
        marks_b=(3, 5)
    )
    
    PATTERNS[(subject, ExamType.MID2_3_4_5)] = QuestionPattern(
        part_a={4: {'short': 2}, 5: {'short': 2}, 6: {'short': 1}},
        part_b={4: {'long': 2}, 5: {'long': 2}, 6: {'long': 2}},
        marks_a=(5, 2),
        marks_b=(3, 5)
    )
    
    PATTERNS[(subject, ExamType.MID2_5_5)] = QuestionPattern(
        part_a={4: {'short': 2}, 5: {'short': 2}, 6: {'short': 1}},
        part_b={4: {'long': 2}, 5: {'long': 2}, 6: {'long': 2}},
        marks_a=(5, 2),
        marks_b=(3, 5)
    )
    
    PATTERNS[(subject, ExamType.REGULAR)] = QuestionPattern(
        part_a={1: {'short': 2}, 2: {'short': 2}, 3: {'short': 1}, 
               4: {'short': 1}, 5: {'short': 2}, 6: {'short': 2}},
        part_b={1: {'long': 2}, 2: {'long': 2}, 3: {'long': 2}, 
               4: {'long': 2}, 5: {'long': 2}, 6: {'long': 2}},
        marks_a=(10, 1),
        marks_b=(6, 10)
    )

def add_eg_patterns():
    """Add patterns for EG"""
    PATTERNS[(SubjectType.EG, ExamType.MID1_2)] = QuestionPattern(
        part_a={},  # No Part A for EG
        part_b={1: {'long': 3}, 2: {'long': 3}},
        marks_a=(0, 0),
        marks_b=(3, 10)
    )
    
    PATTERNS[(SubjectType.EG, ExamType.MID1_2_5)] = QuestionPattern(
        part_a={},  # No Part A for EG
        part_b={1: {'long': 2}, 2: {'long': 2}, 3: {'long': 2}},
        marks_a=(0, 0),
        marks_b=(3, 10)
    )
    
    PATTERNS[(SubjectType.EG, ExamType.MID2_3_4_5)] = QuestionPattern(
        part_a={},  # No Part A for EG
        part_b={4: {'long': 2}, 5: {'long': 2}, 6: {'long': 2}},
        marks_a=(0, 0),
        marks_b=(3, 10)
    )
    
    PATTERNS[(SubjectType.EG, ExamType.MID2_5_5)] = QuestionPattern(
        part_a={},  # No Part A for EG
        part_b={4: {'long': 2}, 5: {'long': 2}, 6: {'long': 2}},
        marks_a=(0, 0),
        marks_b=(3, 10)
    )
    
    PATTERNS[(SubjectType.EG, ExamType.REGULAR)] = QuestionPattern(
        part_a={},  # No Part A for EG
        part_b={1: {'long': 2}, 2: {'long': 2}, 3: {'long': 2}, 
               4: {'long': 2}, 5: {'long': 2}},
        marks_a=(0, 0),
        marks_b=(5, 14)
    )

# Initialize patterns for all subjects
for subject in SubjectType:
    if subject == SubjectType.EG:
        add_eg_patterns()
    elif subject in [SubjectType.BCME, SubjectType.BEEE]:
        add_bcme_beee_patterns(subject)
    else:
        add_general_subject_patterns(subject)

# Add supply patterns (same as regular)
for subject in SubjectType:
    if (subject, ExamType.REGULAR) in PATTERNS:
        PATTERNS[(subject, ExamType.SUPPLY)] = PATTERNS[(subject, ExamType.REGULAR)]

def get_google_drive_file_id(url: str) -> str:
    """Extract file ID from Google Drive URL"""
    patterns = [
        r'https://drive\.google\.com/file/d/(.*?)/view',
        r'https://drive\.google\.com/open\?id=(.*)',
        r'https://drive\.google\.com/uc\?id=(.*?)&'
    ]
    
    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            return match.group(1)
    return None

def get_image_from_url(url: str) -> BytesIO:
    """Download image from any URL"""
    try:
        file_id = get_google_drive_file_id(url)
        if file_id:
            download_url = f"https://drive.google.com/uc?id={file_id}"
        else:
            download_url = url
            
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        
        response = requests.get(download_url, headers=headers, timeout=10)
        response.raise_for_status()
        
        if not response.headers.get('content-type', '').startswith('image/'):
            raise ValueError("URL does not point to a valid image")
            
        return BytesIO(response.content)
    except Exception as e:
        raise Exception(f"Error downloading image: {str(e)}")

def configure_question_image(img: Image, max_width: int = 250, max_height: int = 400) -> tuple:
    """Configure question image dimensions while maintaining aspect ratio"""
    width, height = img.size
    aspect_ratio = width / height
    
    if width > max_width or height > max_height:
        if width / max_width > height / max_height:
            new_width = max_width
            new_height = int(max_width / aspect_ratio)
        else:
            new_height = max_height
            new_width = int(max_height * aspect_ratio)
    else:
        new_width, new_height = width, height
        
    return new_width, new_height

def extract_question_and_url(text: str) -> tuple:
    """Extract question text and image URL from multi-line text"""
    if pd.isna(text):
        return "", None
        
    lines = str(text).split('\n')
    image_url = None
    
    if lines and any(line.strip().startswith(('http://', 'https://')) for line in lines):
        for line in reversed(lines):
            if line.strip().startswith(('http://', 'https://')):
                image_url = line.strip()
                lines = [l for l in lines if l != line]
                break
                
    question = '\n'.join(lines).strip()
    return question, image_url

def process_math_formatting(text: str) -> str:
    """Convert Excel-style subscripts and superscripts to ReportLab-compatible formatting"""
    if pd.isna(text):
        return ""
    
    # Pre-process text to normalize spaces
    text = text.strip()
    
    # First pass: Handle explicit unicode superscripts/subscripts
    superscript_map = {
        '²': '<sup>2</sup>',
        '³': '<sup>3</sup>',
        '⁴': '<sup>4</sup>',
        '⁵': '<sup>5</sup>',
        '⁶': '<sup>6</sup>',
        '⁷': '<sup>7</sup>',
        '⁸': '<sup>8</sup>',
        '⁹': '<sup>9</sup>'
    }
    subscript_map = {
        '₀': '<sub>0</sub>', '₁': '<sub>1</sub>', '₂': '<sub>2</sub>',
        '₃': '<sub>3</sub>', '₄': '<sub>4</sub>', '₅': '<sub>5</sub>',
        '₆': '<sub>6</sub>', '₇': '<sub>7</sub>', '₈': '<sub>8</sub>',
        '₉': '<sub>9</sub>'
    }
    
    for sup, repl in superscript_map.items():
        text = text.replace(sup, repl)
    for sub, repl in subscript_map.items():
        text = text.replace(sub, repl)

    # First identify chemistry formulas vs math equations
    def is_chemistry_formula(string):
        # Check for common patterns in chemistry formulas like N2, O2, H2O, etc.
        chemistry_pattern = r'\b[A-Z][a-z]?\d+\b'
        chemistry_keywords = ["molecular", "molecule", "orbital", "Bond", "magnetic"]
        return bool(re.search(chemistry_pattern, string)) and any(keyword in string for keyword in chemistry_keywords)

    if is_chemistry_formula(text):
        # Handle chemistry formulas (make numbers subscript)
        text = re.sub(r'([A-Z][a-z]?)(\d+)', r'\1<sub>\2</sub>', text)
    else:
        # Handle mathematical expressions (make all numbers after letters superscript)
        text = re.sub(r'([a-zA-Z])(\d+)(?=[\+\-\s]|$|\)|\Z)', r'\1<sup>\2</sup>', text)
    
    # Handle special characters and symbols
    replacements = {
        '°': '&#176;',  # degree symbol
        '±': '&#177;',  # plus-minus
        '→': '&#8594;', # right arrow
        '←': '&#8592;', # left arrow
        'α': '&#945;',  # alpha
        'β': '&#946;',  # beta
        'γ': '&#947;',  # gamma
        'Δ': '&#916;',  # delta
        'π': '&#960;',  # pi
        '∞': '&#8734;', # infinity
    }
    
    for old, new in replacements.items():
        text = text.replace(old, new)
    
    return text

class CustomDocTemplate(SimpleDocTemplate):
    def __init__(self, *args, **kwargs):
        self.top_margin_first = kwargs.pop('top_margin_first', 0)
        self.top_margin_rest = kwargs.pop('top_margin_rest', 30)
        self.encryption = kwargs.pop('encryption', None)
        kwargs['topMargin'] = self.top_margin_first
        
        if self.encryption:
            kwargs['encrypt'] = self.encryption
            
        SimpleDocTemplate.__init__(self, *args, **kwargs)
        
    def handle_pageBegin(self):
        """Modify top margin based on page number"""
        self._calc = self.canv.getPageNumber() > 1
        self.topMargin = self.top_margin_rest if self._calc else self.top_margin_first
        return super().handle_pageBegin()

def create_header_config():
    """Create and get header configuration from user input"""
    st.markdown('<div class="section-header"><h2>📝 Paper Configuration</h2></div>', 
                unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        academic_level = st.selectbox("Academic Level", ACADEMIC_LEVELS)
        semester = st.selectbox("Semester", ["I", "II"])
    
    with col2:
        exam_type = st.selectbox(
            "Exam Type",
            ["I MID", "II MID", "REGULAR", "SUPPLY"]
        )
        branches = st.multiselect(
            "Branch",
            BRANCHES,
            default=["CSE", "CSE(AIML)"]
        )
    
    with col3:
        month = st.selectbox("Month", MONTHS)
        year = st.selectbox("Year", range(2024, 2030))
    
    return {
        'academic_level': academic_level,
        'semester': semester,
        'exam_type': exam_type,
        'month_year': f"{month}-{year}",
        'branches': branches,
        'date': datetime.now().strftime("%d %B %Y"),
    }

def validate_question_counts(df: pd.DataFrame, subject: SubjectType, exam: ExamType) -> list:
    """Validate if enough questions are available for the selected pattern"""
    errors = []
    
    required_columns = ['sno', 'unit', 'question', 'type', 'CO', 'PO', 'BTL']
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        return [f"Missing required columns: {', '.join(missing_columns)}"]
    
    pattern = PATTERNS.get((subject, exam))
    if not pattern:
        return [f"No pattern defined for {subject.value} - {exam.value}"]

    # Check Part A
    for unit, type_counts in pattern.part_a.items():
        for qtype, count in type_counts.items():
            available = len(df[(df['unit'] == unit) & (df['type'] == qtype)])
            if available < count:
                errors.append(f"Unit {unit} needs {count} {qtype} questions for Part A, but has only {available}")

    # Check Part B
    for unit, type_counts in pattern.part_b.items():
        for qtype, count in type_counts.items():
            available = len(df[(df['unit'] == unit) & (df['type'] == qtype)])
            if available < count:
                errors.append(f"Unit {unit} needs {count} {qtype} questions for Part B, but has only {available}")

    return errors

def select_questions(df: pd.DataFrame, subject: SubjectType, exam: ExamType) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Select questions according to the pattern"""
    pattern = PATTERNS[(subject, exam)]
    
    # Select Part A questions
    part_a = pd.DataFrame()
    for unit, type_counts in pattern.part_a.items():
        for qtype, count in type_counts.items():
            questions = df[
                (df['unit'] == unit) & 
                (df['type'] == qtype)
            ].sample(n=count)
            part_a = pd.concat([part_a, questions])
    
    # Select Part B questions
    part_b = pd.DataFrame()
    for unit, type_counts in pattern.part_b.items():
        for qtype, count in type_counts.items():
            questions = df[
                (df['unit'] == unit) & 
                (df['type'] == qtype)
            ].sample(n=count)
            part_b = pd.concat([part_b, questions])
    
    return part_a, part_b

def generate_pdf_with_header(part_a: pd.DataFrame, part_b: pd.DataFrame, 
                           subject: SubjectType, exam: ExamType, 
                           header_info: dict, college_logo_url: str,
                           max_image_width: int = 410) -> bytes:
    """Generate PDF with header and properly scaled images"""
    buffer = BytesIO()
    
    # Create encryption object
    encryption = StandardEncryption(
        userPassword='a'.encode('utf-8'),
        ownerPassword='admin123'.encode('utf-8'),
        strength=128,
        canPrint=1,
        canModify=0,
        canCopy=0,
        canAnnotate=0
    )
    
    # Initialize document with correct margins and encryption
    doc = CustomDocTemplate(
        buffer,
        pagesize=A4,
        bottomMargin=30,
        leftMargin=30,
        rightMargin=30,
        top_margin_first=0,
        top_margin_rest=30,
        encryption=encryption
    )
    
    styles = getSampleStyleSheet()
    story = []

    # Custom styles
    styles.add(ParagraphStyle(
        'QuestionStyle',
        parent=styles['Normal'],
        fontSize=11,
        leading=14,
        spaceBefore=6,
        spaceAfter=6,
        wordWrap='CJK',
        allowWidows=1,
        allowOrphans=1
    ))

    styles.add(ParagraphStyle(
        'CollegeHeader',
        parent=styles['Title'],
        fontSize=16,
        alignment=1,
        spaceAfter=2
    ))
    
    styles.add(ParagraphStyle(
        'ExamHeader',
        parent=styles['Title'],
        fontSize=14,
        alignment=1,
        spaceAfter=2
    ))
    
    styles.add(ParagraphStyle(
        'DetailsLeft',
        parent=styles['Normal'],
        fontSize=11,
        alignment=0,
        leftIndent=20
    ))
    
    styles.add(ParagraphStyle(
        'DetailsRight',
        parent=styles['Normal'],
        fontSize=11,
        alignment=2,
        rightIndent=20
    ))
    
    styles.add(ParagraphStyle(
        'SectionHeader',
        parent=styles['Normal'],
        fontSize=12,
        fontName='Helvetica-Bold',
        alignment=1,
        spaceBefore=6,
        spaceAfter=6
    ))

    def create_question_cell(question_text: str, image_url: str = None) -> List:
        elements = []
        if question_text:
            formatted_text = process_math_formatting(question_text)
            text = Paragraph(formatted_text, styles['QuestionStyle'])
            elements.append(text)

        if image_url:
            try:
                img_data = get_image_from_url(image_url)
                img = Image.open(img_data)
                
                if img.mode == 'RGBA':
                    background = Image.new('RGB', img.size, (255, 255, 255))
                    background.paste(img, mask=img.split()[3])
                    img = background
                elif img.mode not in ['RGB', 'L']:
                    img = img.convert('RGB')
                
                width, height = configure_question_image(img, max_image_width)
                
                img_byte_arr = BytesIO()
                img.save(img_byte_arr, format='JPEG', quality=85, optimize=True)
                img_byte_arr = img_byte_arr.getvalue()
                
                img_reader = BytesIO(img_byte_arr)
                elements.append(Spacer(1, 6))
                elements.append(RLImage(img_reader, width=width, height=height))
            except Exception as e:
                elements.append(Paragraph(f"[Error loading image: {str(e)}]", styles['Normal']))

        return elements

    # Add the logo
    default_logo_url = "https://drive.google.com/file/d/11cfL6HFoSRsCcFWdSzwNuLibTyvASLk8/view?usp=drive_link"
    logo_url_to_use = college_logo_url if college_logo_url and college_logo_url.strip() else default_logo_url

    try:
        img_data = get_image_from_url(logo_url_to_use)
        img = Image.open(img_data)
        
        if img.mode == 'RGBA':
            background = Image.new('RGB', img.size, (255, 255, 255))
            background.paste(img, mask=img.split()[3])
            img = background
        elif img.mode not in ['RGB', 'L']:
            img = img.convert('RGB')
            
        aspect = img.width / img.height
        desired_width = 540
        height = int(desired_width / aspect)
        
        img_byte_arr = BytesIO()
        img.save(img_byte_arr, format='JPEG', quality=85, optimize=True)
        img_byte_arr = img_byte_arr.getvalue()
        
        img_reader = BytesIO(img_byte_arr)
        logo = RLImage(img_reader, width=desired_width, height=height)
        story.append(logo)
    except Exception as e:
        story.append(Spacer(1, 36))

    # Header content
    story.append(Spacer(1, 2))
    story.append(HRFlowable(width="100%", thickness=1, color=colors.black))
    story.append(Spacer(1, 2))

    exam_header = (
        f"{header_info['academic_level']} {header_info['semester']} SEMESTER "
        f"{header_info['exam_type']} EXAMINATION {header_info['month_year']}"
    )
    story.append(Paragraph(exam_header, styles['ExamHeader']))

    # Determine time and marks
    exam_time = "2 Hours" if header_info['exam_type'] in ["I MID", "II MID"] else "3 Hours"
    max_marks = "25M" if header_info['exam_type'] in ["I MID", "II MID"] else "70M"

    # Create table for details
    branch_text = f"Branch: {', '.join(header_info['branches'])}"
    date_text = f"Date: {header_info['date']}"
    subject_text = f"Subject: {subject.value}"
    
    data = [
        [Paragraph(branch_text, styles['DetailsLeft']), 
         Paragraph("Regulation: RV23", styles['DetailsRight'])],
        [Paragraph(date_text, styles['DetailsLeft']), 
         Paragraph(f"Max.Marks: {max_marks}", styles['DetailsRight'])],
        [Paragraph(subject_text, styles['DetailsLeft']), 
         Paragraph(f"Time: {exam_time}", styles['DetailsRight'])]
    ]

    # Calculate column widths
    total_width = doc.width
    col_widths = [total_width * 0.7, total_width * 0.3]
    details_table = Table(data, colWidths=col_widths)
    details_table.setStyle(TableStyle([
        ('ALIGN', (0, 0), (0, -1), 'LEFT'),
        ('ALIGN', (-1, 0), (-1, -1), 'RIGHT'),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('LEFTPADDING', (0, 0), (-1, -1), 0),
        ('RIGHTPADDING', (0, 0), (-1, -1), 0),
        ('TOPPADDING', (0, 0), (-1, -1), 3),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 3),
    ]))
    
    story.append(details_table)
    story.append(Spacer(1, 2))
    # story.append(HRFlowable(width="100%", thickness=1, color=colors.black))
    story.append(Spacer(1, 2))

    # Define column widths for question table
    total_width = doc.width
    col_widths = [
        total_width * 0.06,  # Q.No
        total_width * 0.79,  # Question
        total_width * 0.05,  # CO
        total_width * 0.05,  # PO
        total_width * 0.05   # BTL
    ]

    # Create table headers
    headers = ['Q.No', 'Question', 'CO', 'PO', 'BTL']
    table_data = [headers]
    pattern = PATTERNS.get((subject, exam))

    def add_question_to_table(q_num, question_row):
        """Add question to table with enhanced formatting"""
        if isinstance(question_row, pd.Series):
            question, image_url = extract_question_and_url(question_row['question'])
            co = str(question_row['CO']) if pd.notna(question_row['CO']) else ""
            po = str(question_row['PO']) if pd.notna(question_row['PO']) else ""
            btl = str(question_row['BTL']) if pd.notna(question_row['BTL']) else ""
        else:
            question, image_url = extract_question_and_url(question_row.question)
            co = str(question_row.CO) if hasattr(question_row, 'CO') and pd.notna(question_row.CO) else ""
            po = str(question_row.PO) if hasattr(question_row, 'PO') and pd.notna(question_row.PO) else ""
            btl = str(question_row.BTL) if hasattr(question_row, 'BTL') and pd.notna(question_row.BTL) else ""

        cell_content = create_question_cell(question, image_url)
        return [str(q_num), cell_content, co, po, btl]

    if subject == SubjectType.EG:
        q_num = 1
        questions = pd.concat([part_a, part_b]) if not part_a.empty else part_b
        
        for i in range(0, len(questions), 2):
            if i+1 < len(questions):
                table_data.append(add_question_to_table(q_num, questions.iloc[i]))
                table_data.append(['', Paragraph('<b>(OR)</b>', styles['QuestionStyle']), '', '', ''])
                table_data.append(add_question_to_table(q_num+1, questions.iloc[i+1]))
                q_num += 2
    else:
        # Part A with bold header
        if not part_a.empty:
            marks_text = f"<b>PART-A :   ANSWER ALL QUESTIONS  :  ({pattern.marks_a[0]}×{pattern.marks_a[1]}={pattern.marks_a[0]*pattern.marks_a[1]}M)</b>"
            table_data.append(['', Paragraph(marks_text, styles['SectionHeader']), '', '', ''])
            part_a_header_row = len(table_data) - 1
            
            sorted_part_a = part_a.sort_values(['unit', 'question'])
            for idx, row in enumerate(sorted_part_a.itertuples()):
                question_letter = f"1.{chr(97+idx)})"
                table_data.append(add_question_to_table(question_letter, row))

        # Part B with bold header
        if not part_b.empty:
            marks_text = f"<b>PART-B : ANSWER ONE QUESTION FROM EACH UNIT : ({pattern.marks_b[0]}×{pattern.marks_b[1]}={pattern.marks_b[0]*pattern.marks_b[1]}M)</b>"
            table_data.append(['', Paragraph(marks_text, styles['SectionHeader']), '', '', ''])
            part_b_header_row = len(table_data) - 1
            
            sorted_part_b = part_b.sort_values('unit')
            q_num = 2
            for i in range(0, len(sorted_part_b), 2):
                if i+1 < len(sorted_part_b):
                    table_data.append(add_question_to_table(q_num, sorted_part_b.iloc[i]))
                    table_data.append(['', Paragraph('<b>(OR)</b>', styles['QuestionStyle']), '', '', ''])
                    table_data.append(add_question_to_table(q_num+1, sorted_part_b.iloc[i+1]))
                    q_num += 2

    # Create and style the table
    question_table = Table(table_data, colWidths=col_widths, repeatRows=1)
    table_style = [
        ('BOX', (0, 0), (-1, -1), 1.0, colors.black),
        ('INNERGRID', (0, 0), (-1, -1), 0.5, colors.black),
        
        # Alignment styles
        ('ALIGN', (0, 0), (-1, 0), 'CENTER'),
        ('ALIGN', (0, 1), (0, -1), 'CENTER'),
        ('ALIGN', (2, 1), (-1, -1), 'CENTER'),
        ('VALIGN', (0, 0), (-1, -1), 'TOP'),
        
        # Font styles
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 11),
        
        # Padding
        ('TOPPADDING', (0, 0), (-1, -1), 6),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
        ('LEFTPADDING', (0, 0), (-1, -1), 6),
        ('RIGHTPADDING', (0, 0), (-1, -1), 6),
        
        # Header background
        ('BACKGROUND', (0, 0), (-1, 0), colors.lightgrey),
        ('LINEBELOW', (0, 0), (-1, 0), 1.5, colors.black),
    ]

    # Add background color for PART-A and PART-B headers if not EG subject
    if subject != SubjectType.EG:
        if not part_a.empty:
            table_style.append(('BACKGROUND', (0, part_a_header_row), (-1, part_a_header_row), colors.lightgrey))
        if not part_b.empty:
            table_style.append(('BACKGROUND', (0, part_b_header_row), (-1, part_b_header_row), colors.lightgrey))

    question_table.setStyle(TableStyle(table_style))
    story.append(question_table)
    doc.build(story)
    buffer.seek(0)
    return buffer.getvalue()

def check_password():
    """Returns `True` if the user had the correct password."""

    def password_entered():
        """Checks whether a password entered by the user is correct."""
        if hmac.compare_digest(st.session_state["username"].strip(), "a") and \
           hmac.compare_digest(st.session_state["password"].strip(), "a"):
            st.session_state["password_correct"] = True
            del st.session_state["password"]  # Don't store password
            del st.session_state["username"]  # Don't store username
        else:
            st.session_state["password_correct"] = False

    if "password_correct" not in st.session_state:
        # First run, show input for username and password
        st.markdown('<div class="main-header"><h1>🔐 Login Required</h1></div>', 
                    unsafe_allow_html=True)
        st.text_input("Username", key="username")
        st.text_input("Password", type="password", key="password")
        st.button("Log In", on_click=password_entered)
        return False
    
    elif not st.session_state["password_correct"]:
        # Password incorrect, show input + error
        st.markdown('<div class="main-header"><h1>🔐 Login Required</h1></div>', 
                    unsafe_allow_html=True)
        st.text_input("Username", key="username")
        st.text_input("Password", type="password", key="password")
        st.error("😕 Invalid credentials")
        st.button("Log In", on_click=password_entered)
        return False
    else:
        # Password correct
        return True

def main():
    if not check_password():
        st.stop()

    st.markdown('<div class="main-header"><h1>📝 Question Paper Generator</h1></div>', 
                unsafe_allow_html=True)

    with st.expander("ℹ️ Instructions", expanded=False):
        st.markdown("""
        ### How to use:
        1. Configure the paper header details
        2. Upload your question bank Excel file
        3. Select subject and pattern
        4. Generate and download your question paper
        
        ### Required Excel columns:
        - sno
        - unit
        - question
        - type
        - CO
        - PO
        - BTL
        
        ### For questions with images:
        - Write question text
        - Press Alt+Enter
        - Add image URL on new line
        """)

    # Add college logo URL input
    default_logo_url = "https://drive.google.com/file/d/11cfL6HFoSRsCcFWdSzwNuLibTyvASLk8/view?usp=drive_link"
    college_logo_url = st.text_input(
        "College Logo URL",
        value=default_logo_url,
        help="Enter the URL for the college logo image"
    )
    
    if not college_logo_url.strip():
        college_logo_url = default_logo_url
    
    header_info = create_header_config()
    
    uploaded_file = st.file_uploader(
        "📤 Upload Question Bank (Excel)",
        type=['xlsx'],
        help="Upload your question bank in Excel format"
    )
    
    if uploaded_file:
        try:
            # Read the Excel file
            df = pd.read_excel(uploaded_file)
            
            # Clean up and preprocess the data
            if 'BTL' in df.columns:
                # Convert BTL column to string type to ensure consistency
                df['BTL'] = df['BTL'].astype(str)
                # Remove any whitespace
                df['BTL'] = df['BTL'].str.strip()
            
            # Similarly handle CO and PO columns if they exist
            if 'CO' in df.columns:
                df['CO'] = df['CO'].astype(str)
                df['CO'] = df['CO'].str.strip()
            
            if 'PO' in df.columns:
                df['PO'] = df['PO'].astype(str)
                df['PO'] = df['PO'].str.strip()
            
            # Show question bank overview
            st.subheader("📊 Question Bank Overview")
            overview = df.pivot_table(
                index='unit',
                columns='type',
                values='sno',
                aggfunc='count',
                fill_value=0
            )
            st.dataframe(overview, use_container_width=True)
            
            # Subject and pattern selection
            col1, col2 = st.columns(2)
            with col1:
                available_subjects = SEMESTER_SUBJECTS.get(
                    (header_info['academic_level'], header_info['semester']), 
                    []
                )
                if not available_subjects:
                    st.error(f"No subjects defined for {header_info['academic_level']} {header_info['semester']} Semester")
                    return

                subject = st.selectbox(
                    "Select Subject",
                    [s.value for s in available_subjects]
                )
                subject = next(s for s in SubjectType if s.value == subject)
            
            with col2:
                if header_info['exam_type'] == "I MID":
                    exam_options = [ExamType.MID1_2.value, ExamType.MID1_2_5.value]
                elif header_info['exam_type'] == "II MID":
                    exam_options = [
                        ExamType.MID2_3_4_5.value,
                        ExamType.MID2_5_5.value
                    ]
                else:
                    exam_options = [ExamType.REGULAR.value, ExamType.SUPPLY.value]
                
                exam = ExamType(st.selectbox(
                    "Select Question Pattern",
                    exam_options
                ))
            
            if st.button("🎯 Generate Question Paper", use_container_width=True):
                try:
                    # Validate question counts
                    errors = validate_question_counts(df, subject, exam)
                    if errors:
                        st.error("Insufficient questions:")
                        for error in errors:
                            st.write(f"- {error}")
                        return
                    
                    # Select questions
                    part_a, part_b = select_questions(df, subject, exam)
                    
                    # Create tabs
                    tab1, tab2 = st.tabs(["Selected Questions", "Download Options"])
                    
                    with tab1:
                        st.subheader("Selected Questions Overview")
                        selected_questions = pd.concat([
                            part_a.assign(part='A'),
                            part_b.assign(part='B')
                        ])
                        
                        # Clean up display dataframe
                        display_df = selected_questions.copy()
                        display_df['question'] = display_df['question'].apply(
                            lambda x: extract_question_and_url(x)[0]
                        )
                        
                        columns_order = ['part', 'unit', 'type', 'question', 'CO', 'PO', 'BTL']
                        display_columns = [col for col in columns_order if col in display_df.columns]
                        display_df = display_df[display_columns]
                        
                        display_df.columns = [col.title() for col in display_df.columns]
                        
                        st.dataframe(
                            display_df,
                            use_container_width=True,
                            hide_index=True
                        )
                        
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Total Questions", len(display_df))
                        with col2:
                            st.metric("Part A Questions", len(display_df[display_df['Part'] == 'A']))
                        with col3:
                            st.metric("Part B Questions", len(display_df[display_df['Part'] == 'B']))
                    
                    with tab2:
                        col1, col2 = st.columns(2)
                        with col1:
                            pdf_bytes = generate_pdf_with_header(
                                part_a, part_b, subject, exam,
                                header_info, college_logo_url
                            )
                            st.download_button(
                                "📄 Download Question Paper (PDF)",
                                pdf_bytes,
                                f"{subject.value}_{exam.value}_question_paper.pdf",
                                "application/pdf"
                            )
                            st.info("📝 PDF Password: a")
                        
                        with col2:
                            st.download_button(
                                "📊 Download Selected Questions (CSV)",
                                selected_questions.to_csv(index=False).encode('utf-8'),
                                f"{subject.value}_{exam.value}_selected_questions.csv",
                                "text/csv",
                                key='download-csv'
                            )
                    
                    st.success("✅ Question paper generated successfully!")
                    
                except Exception as e:
                    st.error(f"Error generating question paper: {str(e)}")
                    
        except Exception as e:
            st.error(f"Error loading file: {str(e)}")
            st.info("Please ensure your Excel file has the correct format")

if __name__ == "__main__":
    main()
