from llama_cpp import Llama
from langdetect import detect
from googletrans import Translator

from logger import logger
from memory import (
    add_to_memory,
    retrieve_memory,
    clear_memory,
    load_memory
)
from whisper import(
    listen_for_activation,
    record_audio_to_wav,
    transcribe_with_whisper_cpp,
    generate_tts,
    play_tts
)


# --- CONSTANTS ---
EXIT_COMAND = ['exit', 'выход']
CLEAR_MEMORY_COMAND = ['forget', 'забудь']
LANGUAGE = 'ru'
MEMORY_TOP = 5


# --- TOOLS INITIALIZATION ---
llm = Llama(
    model_path='../models/llama-2-7b-chat.Q4_K_M.gguf',
    n_ctx=1024,
    n_threads=8,
    verbose=False
)
translator = Translator()


# --- TRANSLATION ---
def translate_text(text, target_language=LANGUAGE):
    translated = translator.translate(text, dest=target_language)
    return translated.text

def detect_language(text):
    return detect(text)

def translate_response(ai_response, user_input, language=None):
    if(language is None):
        user_language = detect_language(user_input)
    else:
        user_language = language
    translated_response = translate_text(ai_response, target_language=user_language)
    return translated_response


# --- ENTRY POINT ---
if __name__ == "__main__":
    load_memory()
    logger.info("The AI chat is ready.")
    while True:
        if(listen_for_activation() == 1):
            #user_input_file = record_audio_to_wav()
            #user_input = transcribe_with_whisper_cpp(user_input_file)
            user_input = input()
            logger.info(f"USER: {user_input}")

            if user_input.lower().strip() in EXIT_COMAND:
                break

            if user_input.lower().strip() in CLEAR_MEMORY_COMAND:
                clear_memory()
                continue

            memories = retrieve_memory(user_input, top_k=MEMORY_TOP)
            if memories:
                logger.info("Restored memory:\n\t" + "\n\t".join(memories))
            context = "\n".join(memories)
            additional_prompt = 'Dont think too much and answer no longer then 2 sentences.'
            prompt = f"{context}[INST]{additional_prompt}\n{user_input}[/INST]"

            output = llm(
                prompt,
                max_tokens=512,
                stop=["</s>"],
                echo=False,
                temperature=0.6,
                top_p=0.9
            )

            ai_response = output['choices'][0]['text'].strip()
            add_to_memory(user_input)
            add_to_memory(ai_response)

            logger.info(f"AI not translated: {ai_response}")
            translated_response = translate_response(ai_response, user_input, language=LANGUAGE)
            logger.info(f"AI: {translated_response}")

            ai_responce_file = generate_tts(translated_response, LANGUAGE)
            play_tts(ai_responce_file)
