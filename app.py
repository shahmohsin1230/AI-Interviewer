import streamlit as st
import sqlite3
import hashlib
import os
import time
import sounddevice as sd
import numpy as np
import wave
import requests
from langdetect import detect
from openai import OpenAI
from elevenlabs import ElevenLabs
import PyPDF2
import docx
from datetime import datetime
import pandas as pd
from io import BytesIO
import json
from dotenv import load_dotenv
import threading
import queue

# Load environment variables
load_dotenv()

# Get API keys from environment variables
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
ELEVEN_API_KEY = os.getenv("ELEVEN_API_KEY")

# Initialize clients
openai_client = OpenAI(api_key=OPENAI_API_KEY)
elevenlabs_client = ElevenLabs(api_key=ELEVEN_API_KEY)

# Audio configuration
SAMPLE_RATE = 44100
CHANNELS = 1
DTYPE = np.int16

# Database setup
def init_database():
    conn = sqlite3.connect('ai_interviewer.db')
    cursor = conn.cursor()
    
    # Create admins table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS admins (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            email TEXT UNIQUE NOT NULL,
            password TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    # Create candidates table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS candidates (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            email TEXT UNIQUE NOT NULL,
            password TEXT NOT NULL,
            cv_content TEXT,
            cv_filename TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    # Create interviews table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS interviews (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            candidate_id INTEGER,
            question TEXT NOT NULL,
            answer TEXT,
            feedback TEXT,
            score INTEGER,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (candidate_id) REFERENCES candidates (id)
        )
    ''')
    
    # Create default admin
    default_admin_email = "email"
    default_admin_password = hash_password("password")
    
    cursor.execute('''
        INSERT OR IGNORE INTO admins (email, password) VALUES (?, ?)
    ''', (default_admin_email, default_admin_password))
    
    conn.commit()
    conn.close()

def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

def verify_admin(email, password):
    conn = sqlite3.connect('ai_interviewer.db')
    cursor = conn.cursor()
    hashed_password = hash_password(password)
    cursor.execute('SELECT id FROM admins WHERE email = ? AND password = ?', (email, hashed_password))
    result = cursor.fetchone()
    conn.close()
    return result is not None

def add_admin(email, password):
    conn = sqlite3.connect('ai_interviewer.db')
    cursor = conn.cursor()
    hashed_password = hash_password(password)
    try:
        cursor.execute('INSERT INTO admins (email, password) VALUES (?, ?)', (email, hashed_password))
        conn.commit()
        conn.close()
        return True
    except sqlite3.IntegrityError:
        conn.close()
        return False

def register_candidate(name, email, password):
    conn = sqlite3.connect('ai_interviewer.db')
    cursor = conn.cursor()
    hashed_password = hash_password(password)
    try:
        cursor.execute('INSERT INTO candidates (name, email, password) VALUES (?, ?, ?)', 
                      (name, email, hashed_password))
        conn.commit()
        candidate_id = cursor.lastrowid
        conn.close()
        return candidate_id
    except sqlite3.IntegrityError:
        conn.close()
        return None

def verify_candidate(email, password):
    conn = sqlite3.connect('ai_interviewer.db')
    cursor = conn.cursor()
    hashed_password = hash_password(password)
    cursor.execute('SELECT id, name FROM candidates WHERE email = ? AND password = ?', (email, hashed_password))
    result = cursor.fetchone()
    conn.close()
    return result

def save_cv_content(candidate_id, content, filename):
    conn = sqlite3.connect('ai_interviewer.db')
    cursor = conn.cursor()
    cursor.execute('UPDATE candidates SET cv_content = ?, cv_filename = ? WHERE id = ?', 
                  (content, filename, candidate_id))
    conn.commit()
    conn.close()

def parse_pdf(file):
    try:
        pdf_reader = PyPDF2.PdfReader(file)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()
        return text
    except Exception as e:
        return f"Error parsing PDF: {str(e)}"

def parse_docx(file):
    try:
        doc = docx.Document(file)
        text = ""
        for paragraph in doc.paragraphs:
            text += paragraph.text + "\n"
        return text
    except Exception as e:
        return f"Error parsing DOCX: {str(e)}"

def generate_questions_from_cv(cv_content):
    try:
        prompt = f"""
        Based on the following CV content, generate 7 interview questions that cover:
        1. Work experience and achievements
        2. Technical skills and projects
        3. Educational background
        4. Problem-solving abilities
        5. Career goals and motivations
        
        CV Content:
        {cv_content}
        
        Return only the questions, one per line, numbered 1-5.
        """
        
        response = openai_client.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=500
        )
        
        questions = response.choices[0].message.content.strip().split('\n')
        return [q.strip() for q in questions if q.strip()]
    except Exception as e:
        return [f"Error generating questions: {str(e)}"]

class ContinuousAudioRecorder:
    def __init__(self, sample_rate=SAMPLE_RATE, channels=CHANNELS):
        self.sample_rate = sample_rate
        self.channels = channels
        self.is_recording = False
        self.audio_data = []
        self.stream = None
        
    def audio_callback(self, indata, frames, time, status):
        """Callback function for audio recording"""
        if self.is_recording:
            self.audio_data.append(indata.copy())
    
    def start_recording(self):
        """Start continuous recording"""
        self.audio_data = []
        self.is_recording = True
        self.stream = sd.InputStream(
            callback=self.audio_callback,
            channels=self.channels,
            samplerate=self.sample_rate,
            dtype=DTYPE
        )
        self.stream.start()
    
    def stop_recording(self):
        """Stop recording and return audio data"""
        self.is_recording = False
        if self.stream:
            self.stream.stop()
            self.stream.close()
        
        if self.audio_data:
            # Concatenate all audio chunks
            audio_array = np.concatenate(self.audio_data, axis=0)
            return audio_array
        return None
    
    def get_duration(self):
        """Get current recording duration in seconds"""
        if self.audio_data:
            total_frames = sum(chunk.shape[0] for chunk in self.audio_data)
            return total_frames / self.sample_rate
        return 0

def save_wav(audio_data, filename, sample_rate=SAMPLE_RATE):
    """Save audio data to WAV file"""
    with wave.open(filename, 'wb') as wf:
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(2)  # 2 bytes for int16
        wf.setframerate(sample_rate)
        wf.writeframes(audio_data.tobytes())

def speech_to_text_elevenlabs(audio_path):
    """Convert speech to text using ElevenLabs API"""
    try:
        url = "https://api.elevenlabs.io/v1/speech-to-text"
        headers = {"xi-api-key": ELEVEN_API_KEY}
        
        with open(audio_path, "rb") as audio_file:
            files = {"file": audio_file}
            data = {
                "model_id": "scribe_v1",
                "language_code": "eng",  # Force English only
            }
            
            response = requests.post(url, headers=headers, files=files, data=data)
        
        if response.status_code == 200:
            response_data = response.json()
            transcription = response_data.get("text", "").strip()
            
            if not transcription:
                return None
                
            # Language detection
            try:
                detected_lang = detect(transcription)
                if detected_lang != "en":
                    st.warning(f"Non-English detected ({detected_lang}). Please speak in English.")
                    return None
            except:
                pass
                
            return transcription
        else:
            st.error(f"API Error: {response.status_code}")
            return None
            
    except Exception as e:
        st.error(f"Speech-to-Text Error: {e}")
        return None

def text_to_speech_elevenlabs(text, voice_id="iTIbX50CCkPCSubMpPY5"):
    """Convert text to speech using ElevenLabs"""
    try:
        audio_stream = elevenlabs_client.text_to_speech.convert(
            text=text,
            voice_id=voice_id,
            model_id="eleven_multilingual_v2",
            voice_settings={"stability": 0.10, "similarity_boost": 0.80, "style_exaggeration": 0.60}
        )
        
        audio_data = b""
        for chunk in audio_stream:
            if isinstance(chunk, bytes):
                audio_data += chunk
        
        return audio_data
    except Exception as e:
        st.error(f"Text-to-Speech Error: {e}")
        return None

def analyze_response_with_gpt(question, answer):
    """Analyze the candidate response with GPT"""
    try:
        system_prompt = """
        You are an experienced hiring manager. Analyze the candidate's response for:
        1. Clarity and coherence (0-20 points)
        2. Relevance to the question (0-20 points)
        3. Depth of knowledge (0-20 points)
        4. Communication skills (0-20 points)
        5. Confidence and professionalism (0-20 points)
        
        Provide:
        1. A total score out of 100
        2. Detailed feedback with strengths and areas for improvement
        
        Format your response as:
        SCORE: [number]/100
        
        FEEDBACK:
        [detailed feedback]
        """
        
        response = openai_client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Question: {question}\n\nAnswer: {answer}"}
            ],
            temperature=0.7
        )
        
        content = response.choices[0].message.content
        
        # Extract score and feedback
        lines = content.split('\n')
        score_line = [line for line in lines if line.startswith('SCORE:')]
        score = 0
        if score_line:
            try:
                score = int(score_line[0].split('/')[0].split(':')[1].strip())
            except:
                score = 0
        
        feedback_start = content.find('FEEDBACK:')
        feedback = content[feedback_start + 9:].strip() if feedback_start != -1 else content
        
        return score, feedback
    except Exception as e:
        return 0, f"Analysis failed: {str(e)}"

def save_interview_response(candidate_id, question, answer, feedback, score):
    conn = sqlite3.connect('ai_interviewer.db')
    cursor = conn.cursor()
    cursor.execute('''
        INSERT INTO interviews (candidate_id, question, answer, feedback, score)
        VALUES (?, ?, ?, ?, ?)
    ''', (candidate_id, question, answer, feedback, score))
    conn.commit()
    conn.close()
def get_all_candidates():
    """Fetch all candidates with their information"""
    conn = sqlite3.connect('ai_interviewer.db')
    cursor = conn.cursor()
    cursor.execute('''
        SELECT id, name, email, cv_filename, created_at FROM candidates
        ORDER BY created_at DESC
    ''')
    results = cursor.fetchall()
    conn.close()
    return results

def get_candidate_average_scores():
    """Get average score for each candidate"""
    conn = sqlite3.connect('ai_interviewer.db')
    cursor = conn.cursor()
    cursor.execute('''
        SELECT c.id, c.name, c.email, 
               ROUND(AVG(CASE WHEN i.score > 0 THEN i.score ELSE NULL END), 1) as avg_score,
               COUNT(i.id) as total_interviews
        FROM candidates c
        LEFT JOIN interviews i ON c.id = i.candidate_id
        GROUP BY c.id, c.name, c.email
        ORDER BY avg_score DESC NULLS LAST
    ''')
    results = cursor.fetchall()
    conn.close()
    return results


def get_candidate_cv(candidate_id):
    """Get candidate CV content and filename"""
    conn = sqlite3.connect('ai_interviewer.db')
    cursor = conn.cursor()
    cursor.execute('SELECT cv_content, cv_filename FROM candidates WHERE id = ?', (candidate_id,))
    result = cursor.fetchone()
    conn.close()
    return result


def get_all_interviews():
    conn = sqlite3.connect('ai_interviewer.db')
    cursor = conn.cursor()
    cursor.execute('''
        SELECT i.id, c.name, c.email, i.question, i.answer, i.feedback, i.score, i.created_at
        FROM interviews i
        JOIN candidates c ON i.candidate_id = c.id
        ORDER BY i.created_at DESC
    ''')
    results = cursor.fetchall()
    conn.close()
    return results

def get_candidate_interviews(candidate_id):
    conn = sqlite3.connect('ai_interviewer.db')
    cursor = conn.cursor()
    cursor.execute('''
        SELECT question, answer, created_at
        FROM interviews
        WHERE candidate_id = ?
        ORDER BY created_at DESC
    ''', (candidate_id,))
    results = cursor.fetchall()
    conn.close()
    return results

def admin_panel():
    st.title("🔧 Admin Panel")
    
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Dashboard", "👥 Add Admin", "📝 All Interviews", "👨‍💼 Candidates"])
    
    with tab1:
        st.subheader("Dashboard Overview")
        
        # Get statistics
        conn = sqlite3.connect('ai_interviewer.db')
        cursor = conn.cursor()
        
        cursor.execute('SELECT COUNT(*) FROM candidates')
        total_candidates = cursor.fetchone()[0]
        
        cursor.execute('SELECT COUNT(*) FROM interviews')
        total_interviews = cursor.fetchone()[0]
        
        cursor.execute('SELECT AVG(score) FROM interviews WHERE score > 0')
        avg_score = cursor.fetchone()[0] or 0
        
        conn.close()
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Candidates", total_candidates)
        with col2:
            st.metric("Total Interviews", total_interviews)
        with col3:
            st.metric("Average Score", f"{avg_score:.1f}/100")
    
    with tab2:
        st.subheader("Add New Admin")
        
        with st.form("add_admin_form"):
            new_admin_email = st.text_input("Admin Email")
            new_admin_password = st.text_input("Admin Password", type="password")
            
            if st.form_submit_button("Add Admin"):
                if new_admin_email and new_admin_password:
                    if add_admin(new_admin_email, new_admin_password):
                        st.success("Admin added successfully!")
                    else:
                        st.error("Email already exists or error occurred.")
                else:
                    st.error("Please fill all fields.")
    
    with tab3:
        st.subheader("All Interview Records")
        
        interviews = get_all_interviews()
        if interviews:
            df = pd.DataFrame(interviews, columns=[
                'ID', 'Candidate Name', 'Email', 'Question', 'Answer', 'Feedback', 'Score', 'Date'
            ])
            st.dataframe(df, use_container_width=True)
            
            # Add a detailed view of interview results
            st.subheader("View Interview Details")
            interview_id = st.selectbox("Select Interview ID", [i[0] for i in interviews])
            
            if interview_id:
                # Find the selected interview
                selected_interview = next((i for i in interviews if i[0] == interview_id), None)
                
                if selected_interview:
                    st.write(f"**Candidate:** {selected_interview[1]} ({selected_interview[2]})")
                    st.write(f"**Date:** {selected_interview[7]}")
                    st.write(f"**Score:** {selected_interview[6]}/100")
                    
                    st.write("**Question:**")
                    st.write(selected_interview[3])
                    
                    st.write("**Answer:**")
                    st.write(selected_interview[4])
                    
                    st.write("**Feedback:**")
                    st.write(selected_interview[5])
        else:
            st.info("No interviews found.")
    
    with tab4:
        st.subheader("Candidate Information")
        
        candidates = get_all_candidates()
        
        if candidates:
            # Add sub-tabs for better organization
            subtab1, subtab2 = st.tabs(["📋 Candidate List", "📊 Candidate Scores"])
            
            with subtab1:
                df = pd.DataFrame(candidates, columns=[
                    'ID', 'Name', 'Email', 'CV Filename', 'Registration Date'
                ])
                
                st.dataframe(df, use_container_width=True)
                
                # CV viewer
                st.subheader("View Candidate CV")
                candidate_id = st.selectbox("Select Candidate ID", [c[0] for c in candidates])
                
                if candidate_id:
                    cv_data = get_candidate_cv(candidate_id)
                    
                    if cv_data and cv_data[0]:  # Check if CV content exists
                        st.write(f"**CV Filename:** {cv_data[1]}")
                        
                        # Display CV content
                        with st.expander("View CV Content"):
                            st.text_area("CV Content", cv_data[0], height=300)
                        
                        # Download button
                        cv_content = cv_data[0]
                        cv_filename = cv_data[1] or "candidate_cv.txt"
                        
                        st.download_button(
                            label="Download CV",
                            data=cv_content,
                            file_name=cv_filename,
                            mime="text/plain"
                        )
                        
                        # Show candidate's interview responses
                        st.subheader("Candidate's Interview Responses")
                        interviews = get_candidate_interviews(candidate_id)
                        
                        if interviews:
                            for i, interview in enumerate(interviews):
                                with st.expander(f"Question {i+1}: {interview[0]}"):
                                    st.write("**Answer:**")
                                    st.write(interview[1])
                                    
                                    # Get the full interview details including feedback and score
                                    conn = sqlite3.connect('ai_interviewer.db')
                                    cursor = conn.cursor()
                                    cursor.execute('''
                                        SELECT feedback, score FROM interviews 
                                        WHERE candidate_id = ? AND question = ?
                                    ''', (candidate_id, interview[0]))
                                    feedback_data = cursor.fetchone()
                                    conn.close()
                                    
                                    if feedback_data:
                                        st.write("**Feedback:**")
                                        st.write(feedback_data[0])
                                        st.write(f"**Score:** {feedback_data[1]}/100")
                        else:
                            st.info("No interview responses found for this candidate.")
                    else:
                        st.warning("No CV found for this candidate.")
            
            with subtab2:
                st.subheader("Individual Candidate Average Scores")
                
                # Get candidate scores
                candidate_scores = get_candidate_average_scores()
                
                if candidate_scores:
                    # Create DataFrame for better display
                    scores_df = pd.DataFrame(candidate_scores, columns=[
                        'ID', 'Name', 'Email', 'Average Score', 'Total Interviews'
                    ])
                    
                    # Display as a nice table
                    st.dataframe(scores_df, use_container_width=True)
                    
                    # Create a bar chart for visualization
                    chart_data = scores_df[scores_df['Average Score'].notna()]
                    
                    if not chart_data.empty:
                        st.subheader("Average Scores Visualization")
                        
                        # Create bar chart
                        chart_data_display = chart_data.set_index('Name')['Average Score']
                        st.bar_chart(chart_data_display)
                        
                        # Show top performers
                        st.subheader("Top Performers")
                        top_performers = chart_data.head(5)
                        
                        for idx, row in top_performers.iterrows():
                            col1, col2, col3 = st.columns([3, 1, 1])
                            with col1:
                                st.write(f"**{row['Name']}** ({row['Email']})")
                            with col2:
                                st.metric("Score", f"{row['Average Score']}/100")
                            with col3:
                                st.metric("Interviews", row['Total Interviews'])
                    else:
                        st.info("No scored interviews found yet.")
                else:
                    st.info("No candidates found.")
        else:
            st.info("No candidates found.")
def check_interview_taken(candidate_id):
    """Check if candidate has already completed all interview questions"""
    conn = sqlite3.connect('ai_interviewer.db')
    cursor = conn.cursor()
    
    # First, get the total number of questions asked to this candidate
    cursor.execute('SELECT COUNT(DISTINCT question) FROM interviews WHERE candidate_id = ?', (candidate_id,))
    question_count = cursor.fetchone()[0]
    expected_questions = 7
    
    conn.close()
    # Return True only if the candidate has answered all expected questions
    return question_count >= expected_questions


def candidate_panel():
    if 'candidate_logged_in' not in st.session_state:
        st.session_state.candidate_logged_in = False
        st.session_state.candidate_id = None
        st.session_state.candidate_name = None
    
    if not st.session_state.candidate_logged_in:
        tab1, tab2 = st.tabs(["🔑 Login", "📝 Register"])
        
        with tab1:
            st.subheader("Candidate Login")
            with st.form("candidate_login_form"):
                email = st.text_input("Email")
                password = st.text_input("Password", type="password")
                
                if st.form_submit_button("Login"):
                    result = verify_candidate(email, password)
                    if result:
                        st.session_state.candidate_logged_in = True
                        st.session_state.candidate_id = result[0]
                        st.session_state.candidate_name = result[1]
                        st.success("Login successful!")
                        st.rerun()
                    else:
                        st.error("Invalid credentials.")
        
        with tab2:
            st.subheader("Candidate Registration")
            with st.form("candidate_register_form"):
                name = st.text_input("Full Name")
                email = st.text_input("Email")
                password = st.text_input("Password", type="password")
                confirm_password = st.text_input("Confirm Password", type="password")
                
                if st.form_submit_button("Register"):
                    if name and email and password and confirm_password:
                        if password == confirm_password:
                            candidate_id = register_candidate(name, email, password)
                            if candidate_id:
                                st.success("Registration successful! Please login.")
                            else:
                                st.error("Email already exists or error occurred.")
                        else:
                            st.error("Passwords do not match.")
                    else:
                        st.error("Please fill all fields.")
    
    else:
        st.title(f"👋 Welcome, {st.session_state.candidate_name}!")
        
        if st.button("Logout"):
            st.session_state.candidate_logged_in = False
            st.session_state.candidate_id = None
            st.session_state.candidate_name = None
            st.rerun()
        
        tab1, tab2= st.tabs(["📄 Upload CV", "🎤 Interview"])
        
        with tab1:
            st.subheader("Upload Your CV")
            
            # Check if CV is already uploaded
            conn = sqlite3.connect('ai_interviewer.db')
            cursor = conn.cursor()
            cursor.execute('SELECT cv_content, cv_filename FROM candidates WHERE id = ?', (st.session_state.candidate_id,))
            cv_result = cursor.fetchone()
            conn.close()
            
            if cv_result and cv_result[0]:  # CV already uploaded
                st.success(f"✅ You have already uploaded your CV: {cv_result[1]}")
                st.info("Please proceed to the Interview section to take your interview.")
                
                # Show the uploaded CV
                with st.expander("View Your Uploaded CV"):
                    st.text_area("CV Content", cv_result[0], height=300)
            else:
                # Allow CV upload
                uploaded_file = st.file_uploader("Choose your CV file", type=['pdf', 'docx'])
                
                if uploaded_file is not None:
                    if st.button("Process CV"):
                        with st.spinner("Processing CV..."):
                            if uploaded_file.type == "application/pdf":
                                cv_content = parse_pdf(uploaded_file)
                            elif uploaded_file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
                                cv_content = parse_docx(uploaded_file)
                            else:
                                st.error("Unsupported file format.")
                                cv_content = None
                            
                            if cv_content:
                                save_cv_content(st.session_state.candidate_id, cv_content, uploaded_file.name)
                                st.success("CV uploaded and processed successfully!")
                                st.session_state.cv_processed = True
                                st.rerun()  # Refresh to show the success message
        
        with tab2:
            st.subheader("AI Interview Session")
            
            # Check if CV is uploaded
            conn = sqlite3.connect('ai_interviewer.db')
            cursor = conn.cursor()
            cursor.execute('SELECT cv_content FROM candidates WHERE id = ?', (st.session_state.candidate_id,))
            result = cursor.fetchone()
            conn.close()
            
            if not result or not result[0]:
                st.warning("Please upload your CV first before starting the interview.")
                return
            
            cv_content = result[0]
            
            # Check if the candidate has already taken an interview
            if check_interview_taken(st.session_state.candidate_id):
                st.warning("You have already completed an interview. Each candidate can only take the interview once.")
                
                # Show previous interview results - MODIFIED to only show questions and answers
                st.subheader("Your Interview Results")
                interviews = get_candidate_interviews(st.session_state.candidate_id)
                if interviews:
                    for i, interview in enumerate(interviews):
                        with st.expander(f"Question {i+1}: {interview[0]}"):
                            st.write("**Your Answer:**")
                            st.write(interview[1])
                return
            
            # Initialize session state for interview
            if 'interview_questions' not in st.session_state:
                st.session_state.interview_questions = []
                st.session_state.current_question_index = 0
                st.session_state.recording = False
                st.session_state.transcription = ""
                st.session_state.recorded_audio = None
                st.session_state.analysis = ""
                st.session_state.questions_generated = False
                st.session_state.audio_recorder = None
            
            # Generate questions if not done
            if not st.session_state.questions_generated:
                with st.spinner("Generating interview questions based on your CV..."):
                    st.session_state.interview_questions = generate_questions_from_cv(cv_content)
                    st.session_state.questions_generated = True
            
            if st.session_state.interview_questions:
                current_q_index = st.session_state.current_question_index
                
                if current_q_index < len(st.session_state.interview_questions):
                    current_question = st.session_state.interview_questions[current_q_index]
                    
                    st.write(f"**Question {current_q_index + 1} of {len(st.session_state.interview_questions)}:**")
                    st.write(current_question)
                    
                    # Auto-speak question when it appears
                    if f'spoken_q_{current_q_index}' not in st.session_state:
                        with st.spinner("AI is speaking the question..."):
                            audio_data = text_to_speech_elevenlabs(current_question)
                            if audio_data:
                                st.audio(audio_data, format='audio/wav', autoplay=True)
                        st.session_state[f'spoken_q_{current_q_index}'] = True
                    
                    # Recording controls
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        if st.button("🎤 Start Recording", disabled=st.session_state.recording):
                            st.session_state.recording = True
                            st.session_state.transcription = ""
                            st.session_state.recorded_audio = None
                            st.session_state.analysis = ""
                            
                            # Initialize audio recorder
                            st.session_state.audio_recorder = ContinuousAudioRecorder()
                            st.session_state.audio_recorder.start_recording()
                            st.rerun()
                    
                    with col2:
                        if st.button("⏹️ Stop Recording", disabled=not st.session_state.recording):
                            if st.session_state.audio_recorder:
                                # Stop recording and get audio data
                                audio_data = st.session_state.audio_recorder.stop_recording()
                                st.session_state.recorded_audio = audio_data
                                st.session_state.audio_recorder = None
                            
                            st.session_state.recording = False
                            st.rerun()
                    
                    with col3:
                        # Show recording duration
                        if st.session_state.recording and st.session_state.audio_recorder:
                            duration = st.session_state.audio_recorder.get_duration()
                            st.metric("Recording Time", f"{int(duration//60):02d}:{int(duration%60):02d}")
                    
                    # Recording status display
                    if st.session_state.recording:
                        st.warning("🔴 Recording in progress... Speak your answer and click 'Stop Recording' when finished!")
                        
                        # Auto-refresh to update the duration display
                        time.sleep(1)
                        st.rerun()
                    
                    # Process recorded audio
                    if st.session_state.recorded_audio is not None and not st.session_state.recording:
                        # Save audio to temporary file
                        temp_file = f"temp_recording_{current_q_index}.wav"
                        save_wav(st.session_state.recorded_audio, temp_file)
                        
                        # Show audio playback
                        st.audio(st.session_state.recorded_audio.tobytes(), format='audio/wav', sample_rate=SAMPLE_RATE)
                        
                        # Process transcription if not already done
                        if not st.session_state.transcription:
                            with st.spinner("Converting speech to text..."):
                                transcription = speech_to_text_elevenlabs(temp_file)
                                if transcription:
                                    st.session_state.transcription = transcription
                        
                        # Display transcription
                        if st.session_state.transcription:
                            st.subheader("Your Answer:")
                            st.write(st.session_state.transcription)
                            
                            col1, col2 = st.columns(2)
                            
                            with col1:
                                if st.button("✅ Submit Answer"):
                                    with st.spinner("Analyzing your response..."):
                                        score, feedback = analyze_response_with_gpt(current_question, st.session_state.transcription)
                                        save_interview_response(
                                            st.session_state.candidate_id,
                                            current_question,
                                            st.session_state.transcription,
                                            feedback,
                                            score
                                        )
                                        
                                        # Clean up temporary file
                                        if os.path.exists(temp_file):
                                            os.remove(temp_file)
                                        
                                        # Move to next question
                                        st.session_state.current_question_index += 1
                                        st.session_state.transcription = ""
                                        st.session_state.recorded_audio = None
                                        st.session_state.analysis = ""
                                        
                                        time.sleep(3)  # Show feedback for 3 seconds
                                        st.rerun()
                            
                            with col2:
                                if st.button("🔄 Re-record Answer"):
                                    # Clean up current recording
                                    st.session_state.transcription = ""
                                    st.session_state.recorded_audio = None
                                    if os.path.exists(temp_file):
                                        os.remove(temp_file)
                                    st.rerun()
                        else:
                            st.error("Could not transcribe audio. Please try recording again.")
                
                else:
                    st.success("🎉 Interview completed!")
                    
                    # Clean up any remaining temporary files
                    for i in range(len(st.session_state.interview_questions)):
                        temp_file = f"temp_recording_{i}.wav"
                        if os.path.exists(temp_file):
                            os.remove(temp_file)
def main():
    st.set_page_config(page_title="AI Interviewer System", page_icon="🤖", layout="wide")
    
    # Initialize database
    init_database()
    
    # Initialize session state
    if 'user_type' not in st.session_state:
        st.session_state.user_type = None
        st.session_state.admin_logged_in = False
    
    # Sidebar for navigation
    with st.sidebar:
        st.title("🤖 AI Interviewer")
        
        if st.session_state.user_type is None:
            st.subheader("Select User Type")
            if st.button("👔 Admin"):
                st.session_state.user_type = "admin"
                st.rerun()
            if st.button("👤 Candidate"):
                st.session_state.user_type = "candidate"
                st.rerun()
        else:
            if st.button("🔙 Back to Home"):
                st.session_state.user_type = None
                st.session_state.admin_logged_in = False
                st.rerun()
    
    # Main content
    if st.session_state.user_type == "admin":
        if not st.session_state.admin_logged_in:
            st.title("🔐 Admin Login")
            
            with st.form("admin_login_form"):
                email = st.text_input("Admin Email")
                password = st.text_input("Password", type="password")
                
                if st.form_submit_button("Login"):
                    if verify_admin(email, password):
                        st.session_state.admin_logged_in = True
                        st.success("Login successful!")
                        st.rerun()
                    else:
                        st.error("Invalid admin credentials.")
        else:
            admin_panel()
    
    elif st.session_state.user_type == "candidate":
        candidate_panel()
    
    else:
        st.title("🤖 AI Interviewer System")
        st.write("Welcome to the AI-powered interview system!")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("👔 For Administrators")
            st.write("- Manage candidates and interviews")
            st.write("- View interview analytics")
            st.write("- Add new admins")
            
        with col2:
            st.subheader("👤 For Candidates")
            st.write("- Upload your CV")
            st.write("- Take AI-powered interviews")
        
        st.info("Please select your user type from the sidebar to continue.")

if __name__ == "__main__":
    main()