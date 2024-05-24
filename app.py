import streamlit as st
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from youtube_transcript_api import YouTubeTranscriptApi
import re
import os

def extract_video_id(url):
    pattern = r'(?:https?:\/\/)?(?:www\.)?(?:youtube\.com\/(?:[^\/\n\s]+\/\S+\/|(?:v|e(?:mbed)?)\/|\S*?[?&]v=)|youtu\.be\/)([a-zA-Z0-9_-]{11})'
    match = re.search(pattern, url)
    if match:
        return match.group(1)
    else:
        return None

def generate_questions(url, question_type, number_of_questions, model_name='llama3-70b-8192', humor_mode=False):
    video_id = extract_video_id(url)
    if not video_id:
        return "Invalid YouTube URL. Please enter a valid URL."

    try:
        output = YouTubeTranscriptApi.get_transcript(video_id)
    except Exception as e:
        return f"Failed to retrieve transcript: {str(e)}"
    transcript_text = ' '.join([segment['text'] for segment in output])

    if question_type == 'MCQ':
        instructions = f"""
        Write {number_of_questions} MCQ questions based on this audio text.
        The audio text would contain a lot of filler noise and irrelevant chat,
        but I want you to focus on the main theme or topic they are addressing.
        The questions must be directly from the audio text and but you can also use prior knowledge to verify the correctness.
        The correct answer must be clear from the 4 available choices, with no room for ambiguity.
        The answer needs to be order randomised in the available choices. The answer
        should either be A, B, C or D. The answer must be mixed up in a thoroughly random answer. Do not let the answers all be B! 
        Add a line break after every choice, so that it appears correct in st.write in streamlit.
        The answers should only be shown at the very bottom of all generated results. It should not be directly below the individual question.
        """
        if humor_mode:
            instructions += "please add one humorous and ridiculous choice per question."
    elif question_type == 'short-answer':
        instructions = f"""
        Write {number_of_questions} short answer questions based on this audio text.
        The audio text would contain a lot of filler noise and irrelevant chat,
        but I want you to focus on the main theme or topic they are addressing.
        The questions must be directly from the audio text and but you can also use prior knowledge to verify the correctness.
        The questions should encourage reflective thinking, to test if the student has truly understood the video
        The answers should only be shown at the very bottom of all generated results. It should not be directly below the individual question.
        """
        if humor_mode:
            instructions += "the final question needs to be a very strange and funny question that will make the student laugh."
    elif question_type == 'true-or-false':
        instructions = f"""
        Write {number_of_questions} true-or-false questions based on this audio text.
        The audio text would contain a lot of filler noise and irrelevant chat,
        but I want you to focus on the main theme or topic they are addressing.
        The questions must be directly from the audio text and but you can also use prior knowledge to verify the correctness.
        The questions must have very clear answer which is either true or false with no ambiguity.
        The answers should only be shown at the very bottom of all generated results. It should not be directly below the individual question.
        """
        if humor_mode:
            instructions += "the final question needs to be a very strange and funny question that will make the student laugh."
    elif question_type == 'fill-in-the-blanks':
        instructions = f"""
        Write {number_of_questions} fill-in-the-blank questions based on this audio text.
        The audio text would contain a lot of filler noise and irrelevant chat,
        but I want you to focus on the main theme or topic they are addressing.
        The questions must be directly from the audio text and but you can also use prior knowledge to verify the correctness.
        The blanks must be placed in such a way that they are intuitive to fill while listening to the audio.
        The answers should only be shown at the very bottom of all generated results. It should not be directly below the individual question.
        """
        if humor_mode:
            instructions += "the final question needs to be a very strange and funny question that will make the student laugh."
    else:
        return "Invalid question type. Please choose either 'MCQ', 'short-answer', 'true-or-false' or 'fill-in-the-blanks'."

    system = "You are a helpful assistant that watches a video, transcribes all its contents, and writes questions for students to verify that they understood the video"

    input_text = f"""
    Audio Text: \n{transcript_text}\n
    Instructions: \n{instructions}
    """

    chat = ChatGroq(temperature=0, model_name=model_name)
    prompt = ChatPromptTemplate.from_messages([("system", system), ("human", "{text}")])
    chain = prompt | chat
    result = chain.invoke({"text": input_text})
    return result.content
# Streamlit appimport streamlit as st

# Streamlit app
st.set_page_config(layout='wide')
st.title("YouTube Quiz Maker")

# Create two columns
left_column, right_column = st.columns([2, 1])

# Left column (main content)
with left_column:
    url_choices = [
        "https://www.youtube.com/watch?v=dQw4w9WgXcQ (Music)",
        "https://www.youtube.com/watch?v=SkVDfaHQwRU (Programming)",
        "https://www.youtube.com/watch?v=50Bda5VKbqA&t=454s (Math)",
        "https://www.youtube.com/watch?v=Yzv0gXqoCkc (Pets)",
        "https://www.youtube.com/watch?v=_0QTpylu1aE (Sports)",
        "Custom URL"
    ]

    url_choice = st.selectbox("Select a YouTube URL:", url_choices)

    if url_choice == "Custom URL":
        custom_url = st.text_input("Enter the custom YouTube URL:")
        if custom_url.strip() == "":
            st.warning("Please enter a valid custom YouTube URL.")
        else:
            url = custom_url
    else:
        url = url_choice

    question_type = st.selectbox("Select question type:", ["MCQ", "short-answer", "true-or-false", "fill-in-the-blanks"])

    number_of_questions = st.number_input("Enter the number of questions:", min_value=1, value=4)

    model_name = st.selectbox("Select LLM Model:", ["llama3-70b-8192", "llama3-8b-8192", "mixtral-8x7b-32768", "gemma-7b-it"])

    humor_mode = st.checkbox("Enable Humor Mode")

    if st.button("Generate Questions"):
        result = generate_questions(url, question_type, number_of_questions, model_name, humor_mode)
        st.text_area("Generated Questions:", value=result, height=600)

# Right column (image)
with right_column:
    st.image("https://i.ibb.co/H2JCc8d/38fd495c-8b00-424c-8029-8d0cfb8a3322.jpg", use_column_width=True)

