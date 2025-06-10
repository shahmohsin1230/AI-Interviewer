# 🤖 AI Interviewer System

An intelligent interview platform that conducts automated interviews using AI-powered speech-to-text, text-to-speech, and response analysis capabilities.

## ✨ Features

### For Candidates
- **User Registration & Authentication** - Secure candidate registration and login
- **CV Upload & Processing** - Support for PDF and DOCX file formats
- **AI-Generated Questions** - Custom interview questions based on uploaded CV
- **Voice-Based Interviews** - Real-time speech-to-text conversion for natural interaction
- **Automated Feedback** - AI-powered response analysis and scoring

### For Administrators
- **Admin Dashboard** - Comprehensive overview of all interviews and candidates
- **Candidate Management** - View candidate profiles, CVs, and interview history
- **Interview Analytics** - Detailed scoring and performance metrics
- **Multi-Admin Support** - Add multiple administrators to the system

## 🛠️ Technology Stack

- **Frontend**: Streamlit
- **Database**: SQLite
- **AI Services**: 
  - OpenAI GPT-4 (Question generation & response analysis)
  - ElevenLabs (Speech-to-text & text-to-speech)
- **Audio Processing**: SoundDevice, NumPy, Wave
- **Document Processing**: PyPDF2, python-docx
- **Additional**: Pandas, LangDetect

## 📋 Prerequisites

- Python 3.8 or higher
- OpenAI API key
- ElevenLabs API key
- Microphone access for audio recording

## 🚀 Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/shahmohsin1230/AI-Interviewer.git
   cd ai-interviewer-system
   ```

2. **Install required packages**
   ```bash
   pip install streamlit sqlite3 hashlib sounddevice numpy wave requests langdetect openai elevenlabs PyPDF2 python-docx pandas python-dotenv
   ```

3. **Set up environment variables**
   
   Create a `.env` file in the root directory:
   ```env
   OPENAI_API_KEY=your_openai_api_key_here
   ELEVEN_API_KEY=your_elevenlabs_api_key_here
   ```

4. **Run the application**
   ```bash
   streamlit run app.py
   ```

## 🔧 Configuration

### Default Admin Account
- **Email**: `email`
- **Password**: `password`

### Audio Settings
- **Sample Rate**: 44,100 Hz
- **Channels**: Mono (1 channel)
- **Format**: 16-bit PCM

## 📱 Usage

### For Candidates

1. **Register** - Create a new candidate account
2. **Upload CV** - Submit your resume in PDF or DOCX format
3. **Take Interview** - Complete the AI-generated interview questions
4. **Voice Responses** - Answer questions using your microphone
5. **Get Feedback** - Receive automated scoring and feedback

### For Administrators

1. **Login** - Use admin credentials to access the dashboard
2. **Monitor Interviews** - View all candidate interviews and responses
3. **Analyze Performance** - Review detailed scoring and feedback
4. **Manage System** - Add new admins and monitor system statistics

## 🗂️ Database Schema

### Tables

- **admins** - Administrator accounts and authentication
- **candidates** - Candidate profiles and CV storage
- **interviews** - Interview questions, responses, and scoring

## 🎯 Key Features Explained

### AI Question Generation
The system analyzes uploaded CVs and generates relevant interview questions covering:
- Work experience and achievements
- Technical skills and projects
- Educational background
- Problem-solving abilities
- Career goals and motivations

### Intelligent Response Analysis
Each candidate response is evaluated on:
- Clarity and coherence (0-20 points)
- Relevance to the question (0-20 points)
- Depth of knowledge (0-20 points)
- Communication skills (0-20 points)
- Confidence and professionalism (0-20 points)

### Real-time Audio Processing
- Continuous audio recording during interviews
- Automatic speech-to-text conversion
- Language detection (English only)
- Audio playback for verification

## 🔒 Security Features

- Password hashing using SHA-256
- Session-based authentication
- Secure database connections
- Input validation and sanitization

## 📊 Analytics & Reporting

- Individual candidate performance tracking
- Average scoring across all interviews
- Interview completion statistics
- Candidate ranking and comparison

## 🚨 Troubleshooting

### Common Issues

1. **Audio Recording Problems**
   - Ensure microphone permissions are granted
   - Check audio device settings
   - Verify SoundDevice installation

2. **API Connection Issues**
   - Verify API keys in `.env` file
   - Check internet connection
   - Confirm API key permissions

3. **File Upload Errors**
   - Ensure CV files are in PDF or DOCX format
   - Check file size limitations
   - Verify file is not corrupted

## 📞 Support

For support and questions:
- Create an issue on GitHub
- Contact the development team
- Check the documentation

---

**Note**: This system requires active internet connection for AI processing and may incur costs based on API usage. Please monitor your API usage and set appropriate limits.