from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
import io
import streamlit as st
import PyPDF2
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from skills import SKILLS
import matplotlib.pyplot as plt

st.set_page_config(page_title="Resume Analyzer", layout="centered")
st.title("📄 Resume Analyzer")
st.write("Upload your resume and compare it with a job description to get match insights.")

# ---------------- PDF TEXT EXTRACTION ----------------
def extract_text_from_pdf(pdf_file):
    reader = PyPDF2.PdfReader(pdf_file)
    text = ""
    for page in reader.pages:
        text += page.extract_text()
    return text.lower()

# ---------------- SKILL EXTRACTION ----------------
def extract_skills(text):
    found_skills = []

    for main_skill, variations in SKILLS.items():
        for variation in variations:
            if variation in text:
                found_skills.append(main_skill)
                break

    return found_skills

# ---------------- MISSING SKILLS ----------------
def get_missing_skills(resume_skills, job_text):
    job_text = job_text.lower()
    job_skills = []

    for skill, variations in SKILLS.items():
        for v in variations:
            if v in job_text:
                job_skills.append(skill)
                break

    missing = [skill for skill in job_skills if skill not in resume_skills]
    return missing


# ---------------- MATCHING ----------------
def calculate_match(resume_text, job_text):
    vectorizer = TfidfVectorizer()
    vectors = vectorizer.fit_transform([resume_text, job_text])
    similarity = cosine_similarity(vectors[0:1], vectors[1:2])
    return round(similarity[0][0] * 100, 2)

# ---------------- ATS SCORE ----------------
def calculate_ats_score(resume_text, job_text):
    resume_words = set(resume_text.split())
    job_words = set(job_text.split())

    if len(job_words) == 0:
        return 0

    matched = resume_words.intersection(job_words)
    score = len(matched) / len(job_words) * 100

    return round(score, 2)

# ---------------- VISUALIZATION ----------------
def show_skill_chart(found_skills):
    skills_count = {skill: 1 for skill in found_skills}  # simple frequency chart
    plt.figure(figsize=(8,4))
    plt.bar(skills_count.keys(), skills_count.values(), color='skyblue')
    plt.xticks(rotation=45, ha='right')
    plt.title("Skills Found in Resume")
    st.pyplot(plt)


# ---------------- PDF REPORT GENERATION ----------------
def generate_pdf_report(found_skills, match_score, ats_score, missing, feedback):
    buffer = io.BytesIO()
    c = canvas.Canvas(buffer, pagesize=letter)

    c.setFont("Helvetica", 12)
    y = 750

    c.drawString(50, y, "Resume Analysis Report")
    y -= 40

    c.drawString(50, y, f"Match Score: {match_score}%")
    y -= 20

    c.drawString(50, y, f"ATS Score: {ats_score}%")
    y -= 20

    c.drawString(50, y, "Extracted Skills:")
    y -= 20
    c.drawString(60, y, ", ".join(found_skills) if found_skills else "None")
    y -= 40

    c.drawString(50, y, "Missing Skills:")
    y -= 20
    c.drawString(60, y, ", ".join(missing) if missing else "None")
    y -= 40

    c.drawString(50, y, "Feedback:")
    y -= 20
    c.drawString(60, y, feedback)

    c.save()
    buffer.seek(0)
    return buffer


# ---------------- AI FEEDBACK ----------------
def ai_feedback(match_score, found_skills):
    feedback = ""
    if match_score > 70:
        feedback = "Excellent match! Your resume aligns well with the job description."
    elif match_score > 40:
        missing = [skill for skill in SKILLS if skill not in found_skills]
        feedback = f"Moderate match. Consider highlighting or learning these skills: {', '.join(missing[:5])}."
    else:
        feedback = "Low match. Strongly consider updating your resume to match the job description."
    return feedback

# ---------------- UI ----------------
uploaded_file = st.file_uploader("Upload Resume (PDF)", type="pdf")
job_description = st.text_area("Paste Job Description")

if uploaded_file and job_description:

    resume_text = extract_text_from_pdf(uploaded_file)

    # Skill extraction
    found_skills = extract_skills(resume_text)

    st.subheader("✅ Extracted Skills")
    if found_skills:
        st.success(" ".join([f"#{s}" for s in found_skills]))
        show_skill_chart(found_skills)
    else:
        st.write("No recognized skills found.")

    # Missing skills
    missing = get_missing_skills(found_skills, job_description)

    st.subheader("❌ Missing Skills")
    st.write(", ".join(missing) if missing else "None")

    # Match score
    match_score = calculate_match(resume_text, job_description.lower())

    st.subheader("📊 Resume Match Score")
    st.progress(match_score / 100)
    st.write(f"{match_score}%")

    # ATS score
    ats_score = calculate_ats_score(resume_text, job_description.lower())

    st.subheader("🤖 ATS Score")
    st.progress(ats_score / 100)
    st.write(f"{ats_score}%")

    # Feedback
    feedback = ai_feedback(match_score, found_skills)
    st.info(feedback)

    # Download report
    pdf_file = generate_pdf_report(found_skills, match_score, ats_score, missing, feedback)

    st.download_button(
        "📥 Download Analysis Report",
        pdf_file,
        "resume_report.pdf",
        "application/pdf"
    )
