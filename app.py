
import gradio as gr
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from youtube_transcript_api import YouTubeTranscriptApi
import os
import re



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

# Create the Gradio app
with gr.Blocks() as app:
    gr.Markdown("# YouTube Quiz Maker")
    
    url_choices = [
        "https://www.youtube.com/watch?v=dQw4w9WgXcQ (Music)",
        "https://www.youtube.com/watch?v=SkVDfaHQwRU (Programming)",
        "https://www.youtube.com/watch?v=50Bda5VKbqA&t=454s (Math)",
        "https://www.youtube.com/watch?v=Yzv0gXqoCkc (Pets)",
        "https://www.youtube.com/watch?v=_0QTpylu1aE (Sports)",
        "Enter custom URL"
    ]
    
    url = gr.Dropdown(choices=url_choices, value=url_choices[0], label="YouTube URL")
    
    question_type = gr.Dropdown(choices=["MCQ", "short-answer", "true-or-false", "fill-in-the-blanks"], value="MCQ", label="Question Type")
    
    number_of_questions = gr.Textbox(lines=1, placeholder="Number of questions", value="4", label="Number of Questions")
    model_name = gr.Dropdown(choices=["llama3-70b-8192", "llama3-8b-8192", "mixtral-8x7b-32768", "gemma-7b-it"], value="llama3-70b-8192", label="LLM Model")
    humor_mode = gr.Checkbox(label="Humor Mode")
    
    generate_button = gr.Button("Generate Questions")
    
    result = gr.Textbox(lines=10, label="Generated Questions")

    generate_button.click(
        fn=generate_questions,
        inputs=[url, question_type, number_of_questions, model_name, humor_mode],
        outputs=result
    )

# Launch the app
app.launch()
