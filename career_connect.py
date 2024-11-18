import streamlit as st
#from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import os
import json
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationChain
#from sklearn.feature_extraction.text import TfidfVectorizer
#from langchain.prompts import PromptTemplate
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

# Configure page title and favicon
st.set_page_config(
    page_title="Career Connect",
    page_icon="🎓",
    layout="wide"
)

# Load environment variables
load_dotenv()

# Function to load data from JSON files
def load_data(filename: str) -> list:
    try:
        with open(filename, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        return []
    except json.JSONDecodeError:
        return []

# Function to save data to JSON files
def save_data(data: dict, filename: str) -> None:
    try:
        existing_data = load_data(filename)
        existing_data.append(data)
        with open(filename, 'w') as f:
            json.dump(existing_data, f, indent=4)
    except Exception as e:
        st.error(f"Error saving data: {e}")


def preprocess_profile(profile: dict) -> str:
    """
    Convert profile details into a single text string for vectorization.

    Args:
        profile (dict): Student or mentor profile dictionary

    Returns:
        str: Concatenated string of relevant profile attributes
    """
    # Combine relevant fields into a single text string
    text_fields = [
        ' '.join(profile.get('subject_preferences', [])),
        profile.get('career_goals', ''),
        profile.get('hobbies_interests', ''),
        ' '.join(profile.get('areas_of_expertise', [])),
        profile.get('mentoring_philosophy', ''),
        profile.get('extracurricular_activities', ''),
        profile.get('professional_title', '')
    ]
    return ' '.join(filter(bool, text_fields))


# Initialize LangChain chatbot
def setup_chatbot():
    if 'conversation_chain' not in st.session_state:
        # Initialize the ChatGroq model
        chat_model = ChatGroq(
            groq_api_key=os.getenv("GROQ_API_KEY"),
            model="llama3-8b-8192",
            temperature=1,
            timeout=None,
            max_retries=2,
            max_tokens=1000
        )

        # Create a conversation memory
        memory = ConversationBufferMemory()

        # Create the conversation chain
        st.session_state.conversation_chain = ConversationChain(
            llm=chat_model,
            memory=memory,
            verbose=True
        )

# Function to get response from LangChain
def get_llm_response(user_input: str) -> str:
    try:
        setup_chatbot()
        response = st.session_state.conversation_chain.predict(input=user_input)
        return response
    except Exception as e:
        return f"Error: {str(e)}"

# Function to calculate cosine similarity
def find_best_match_tfidf(student_profile: dict, mentors: list) -> tuple:
    """
    Find the best mentor match using TF-IDF vectorization and cosine similarity.

    Args:
        student_profile (dict): Selected student's profile
        mentors (list): List of mentor profiles

    Returns:
        tuple: Best matching mentor name and similarity score
    """
    # Prepare text data
    texts = [preprocess_profile(student_profile)] + [preprocess_profile(mentor) for mentor in mentors]

    # Create TF-IDF Vectorizer
    vectorizer = TfidfVectorizer()

    # Fit and transform the texts
    tfidf_matrix = vectorizer.fit_transform(texts)

    # Calculate cosine similarities
    similarities = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:])[0]

    # Find the best match
    best_match_index = np.argmax(similarities)
    best_match_score = similarities[best_match_index]

    return (mentors[best_match_index]['full_name'], best_match_score)


# Initialize session state for chat history
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

# Custom CSS
st.markdown("""
    <style>
    .stApp {
        max-width: 1200px;
        margin: 0 auto;
    }
    .chat-message {
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
    }
    .user-message {
        background-color: #000000;
    }
    .ai-message {
        background-color: #000000;
    }
    </style>
""", unsafe_allow_html=True)

# Placeholder data for initial testing
all_subjects = ["Math", "Science", "Art", "Commerce", "Computer Science", "Languages", "Social Studies", "Others"]
all_expertise = ["Technology", "Business", "Healthcare", "Education", "Arts", "Science", "Engineering", "Others"]

# Streamlit App Setup
st.title("🎓 Career Connect")
st.subheader("AI-powered platform for career counseling")

# Navigation Menu
menu = ["Home", "Student Registration", "Mentor Registration", "Chatbot", "Matching"]
choice = st.sidebar.selectbox("Menu", menu)

# Home Page
if choice == "Home":
    st.header("Welcome to Career Connect! 👋")
    st.write("""
    ### Your AI-Powered Career Guidance Platform

    Career Connect offers:
    * 🎯 Personalized career recommendations
    * 👥 Mentor matching based on interests and skills
    * 📚 Access to curated learning resources
    * 🤖 AI-powered career guidance chatbot
    """)
    col1, col2, col3 = st.columns(3)
    with col1:
        st.info("**AI Career Guide**\n\nGet instant career advice from our AI counselor")
    with col2:
        st.success("**Mentor Matching**\n\nConnect with experienced professionals")
    with col3:
        st.warning("**Resource Hub**\n\nAccess curated learning materials")

# Student Registration Page
elif choice == "Student Registration":
    st.header("📝 Student Registration")
    with st.form("student_form"):
        full_name = st.text_input("Full Name")
        gender = st.selectbox("Gender", ["Male", "Female", "Other"])
        class_level = st.text_input("Class / Grade Level")
        subject_preferences = st.multiselect("Subject Preferences", all_subjects)
        extracurricular_activities = st.text_area("Extracurricular Activities")
        hobbies_interests = st.text_area("Hobbies & Interests outside school")
        achievements = st.text_area("Achievements in Extracurricular Activities")
        career_goals = st.text_area("Career Goals and Aspirations")

        submit = st.form_submit_button("Register")
        if submit and full_name:
            student_data = {
                "full_name": full_name,
                "gender": gender,
                "class_level": class_level,
                "subject_preferences": subject_preferences,
                "extracurricular_activities": extracurricular_activities,
                "hobbies_interests": hobbies_interests,
                "achievements": achievements,
                "career_goals": career_goals
                # Remove the vector creation
                # "vector": student_vector
            }
            save_data(student_data, "students.json")
            st.success(f"Welcome aboard, {full_name}! 🎉")
            st.json(student_data)

# Modify Mentor Registration
elif choice == "Mentor Registration":
    st.header("👨‍🏫 Mentor Registration")
    with st.form("mentor_form"):
        full_name = st.text_input("Full Name")
        gender = st.selectbox("Gender", ["Male", "Female", "Other"])
        age = st.number_input("Age", min_value=18, max_value=100, step=1)
        professional_title = st.text_input("Professional Title")
        years_of_experience = st.number_input("Years of Experience", min_value=0, max_value=50, step=1)
        linkedin_profile = st.text_input("LinkedIn Profile URL")
        areas_of_expertise = st.multiselect("Areas of Expertise", all_expertise)
        preferred_communication = st.selectbox("Preferred Communication Method", ["Email", "Phone", "Video Call", "In-person"])
        mentoring_philosophy = st.text_area("Your Mentoring Philosophy")

        submit = st.form_submit_button("Register")
        if submit and full_name:
            mentor_data = {
                "full_name": full_name,
                "gender": gender,
                "age": age,
                "professional_title": professional_title,
                "years_of_experience": years_of_experience,
                "linkedin_profile": linkedin_profile,
                "areas_of_expertise": areas_of_expertise,
                "preferred_communication": preferred_communication,
                "mentoring_philosophy": mentoring_philosophy
                # Remove the vector creation
                # "vector": mentor_vector
            }
            save_data(mentor_data, "mentors.json")
            st.success(f"Thank you for joining as a mentor, {full_name}! 🎉")
            st.json(mentor_data)

# Chatbot Page
elif choice == "Chatbot":
    st.header("🤖 AI Career Guidance Chatbot")
    st.write("Ask me anything about careers, education, and professional development!")

    # Initialize the chatbot
    setup_chatbot()

    # Chat interface
    chat_container = st.container()
    with chat_container:
        for message in st.session_state.chat_history:
            role = "You" if message["role"] == "user" else "AI"
            css_class = "user-message" if message["role"] == "user" else "ai-message"
            st.markdown(f"""
                <div class="chat-message {css_class}">
                    <b>{role}:</b> {message["content"]}
                </div>
            """, unsafe_allow_html=True)

    # User input
    with st.form(key="chat_form"):
        user_query = st.text_input("Type your question here:", key="user_input")
        submit_button = st.form_submit_button("Ask")

        if submit_button and user_query:
            st.session_state.chat_history.append({"role": "user", "content": user_query})
            response = get_llm_response(user_query)
            st.session_state.chat_history.append({"role": "assistant", "content": response})
            st.rerun()

# Matching Page
elif choice == "Matching":
    st.header("🤝 Student-Mentor Matching")
    st.write(
        "Our AI-powered system matches students with mentors based on interests and goals using advanced TF-IDF vectorization.")

    # Load existing profiles
    students = load_data("students.json")
    mentors = load_data("mentors.json")

    if not students or not mentors:
        st.warning("No registered students or mentors available for matching.")
    else:
        selected_student_index = st.selectbox(
            "Select a student to match",
            range(len(students)),
            format_func=lambda x: students[x]["full_name"]
        )
        selected_student = students[selected_student_index]

        # Match the selected student using TF-IDF
        match = find_best_match_tfidf(selected_student, mentors)
        if match:
            st.success(
                f"Best match for {selected_student['full_name']} is {match[0]} with {match[1]:.2%} compatibility")

            # Optional: Display matched mentor details
            matched_mentor = next(mentor for mentor in mentors if mentor['full_name'] == match[0])
            with st.expander("Matched Mentor Details"):
                st.write(f"**Name:** {matched_mentor['full_name']}")
                st.write(f"**Professional Title:** {matched_mentor.get('professional_title', 'N/A')}")
                st.write(f"**Areas of Expertise:** {', '.join(matched_mentor.get('areas_of_expertise', []))}")
                st.write(f"**Experience:** {matched_mentor.get('years_of_experience', 'N/A')} years")
        else:
            st.error("No suitable match found!")
with st.sidebar:
    st.markdown("---")
    st.info("Developed by Code_Crusaders_BCA\nSmart India Hackathon 2024")
    st.markdown("---")
    st.caption("v1.0.0")
    api_status = "🟢 Online" if os.getenv("GROQ_API_KEY") else "🔴 Offline"
    st.caption(f"API Status: {api_status}")