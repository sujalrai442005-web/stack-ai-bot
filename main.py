# ============================================================
# TELEGRAM AI ASSISTANT
# Features:
# - Normal AI Chat + Memory
# - HC Verma Physics Solver Mode
# - HC Verma Question Photo Solver
# - Vision AI
# - Voice Input + Voice Reply
# - PDF Summarizer
# - Web Search
# - AI Image Generation
# - Teacher / Coder / Friend modes
# ============================================================

from telegram import Update
from telegram.ext import (
    ApplicationBuilder,
    MessageHandler,
    CommandHandler,
    filters,
    ContextTypes,
)
from telegram.constants import ChatAction

from dotenv import load_dotenv
from groq import Groq

import os
import json
import random
import requests
import urllib.parse
import base64
import tempfile
import re

from gtts import gTTS


# ============================================================
# ENVIRONMENT
# ============================================================

load_dotenv()

BOT_TOKEN = os.getenv("BOT_TOKEN")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not BOT_TOKEN:
    raise SystemExit(
        "BOT_TOKEN missing.\n"
        "Create a .env file and add:\n"
        "BOT_TOKEN=your_bot_token"
    )

if not GROQ_API_KEY:
    raise SystemExit(
        "GROQ_API_KEY missing.\n"
        "Add it to your .env file."
    )

client = Groq(api_key=GROQ_API_KEY)


# ============================================================
# MODELS
# ============================================================

CHAT_MODEL = os.getenv(
    "GROQ_CHAT_MODEL",
    "llama-3.3-70b-versatile"
)

VISION_MODEL = os.getenv(
    "GROQ_VISION_MODEL",
    "meta-llama/llama-4-scout-17b-16e-instruct"
)

WHISPER_MODEL = os.getenv(
    "GROQ_WHISPER_MODEL",
    "whisper-large-v3"
)


# ============================================================
# MEMORY
# ============================================================

MEMORY_FILE = "memory.json"

try:
    if os.path.exists(MEMORY_FILE):
        with open(MEMORY_FILE, "r", encoding="utf-8") as file:
            user_memory = json.load(file)

        if not isinstance(user_memory, dict):
            user_memory = {}

    else:
        user_memory = {}

except Exception as e:
    print("MEMORY LOAD ERROR:", e)
    user_memory = {}


bot_messages = {}


def save_memory():
    """Save user memory safely."""

    try:
        temp_file = MEMORY_FILE + ".tmp"

        with open(
            temp_file,
            "w",
            encoding="utf-8"
        ) as file:

            json.dump(
                user_memory,
                file,
                ensure_ascii=False,
                indent=2
            )

        os.replace(temp_file, MEMORY_FILE)

    except Exception as e:
        print("MEMORY SAVE ERROR:", e)


# ============================================================
# USER DATA
# ============================================================

def get_user_data(user_id: str) -> dict:

    if user_id not in user_memory:

        user_memory[user_id] = {
            "history": [],
            "mode": "default",
            "name": None,
            "voice_mode": False,
            "hc_verma_mode": False,
        }

    data = user_memory[user_id]

    if not isinstance(data, dict):

        data = {
            "history": [],
            "mode": "default",
            "name": None,
            "voice_mode": False,
            "hc_verma_mode": False,
        }

        user_memory[user_id] = data

    data.setdefault("history", [])
    data.setdefault("mode", "default")
    data.setdefault("name", None)
    data.setdefault("voice_mode", False)
    data.setdefault("hc_verma_mode", False)

    return data


# ============================================================
# AI MODES
# ============================================================

AI_MODES = {

    "default": {
        "label": "🤖 Default AI",

        "system": """
You are Stack AI Bot.

Rules:
- Talk naturally like a real human assistant.
- Keep replies short and smart.
- Maximum 3-5 lines normally.
- Never give code unless user asks for coding.
- Never write unnecessary essays.
- Avoid robotic AI language.
- Sound friendly and modern.
- Use emojis naturally.
- If user talks casually, reply casually.
- If user talks in Hinglish, reply in Hinglish.
- If user talks in English, reply in English.
"""
    },


    "teacher": {
        "label": "🎓 Teacher Mode",

        "system": """
You are an expert teacher and educator.

Rules:
- Explain concepts clearly.
- Break complex topics into simple steps.
- Use examples and analogies.
- Make learning interesting.
- Use simple language unless technical depth is requested.
- Give useful tips and memory tricks.
- Hinglish input → Hinglish answer.
- English input → English answer.
"""
    },


    "coder": {
        "label": "💻 Coder Mode",

        "system": """
You are an expert software engineer and coding assistant.

Rules:
- Provide clean and correct code.
- Explain the code.
- Point out bugs and improvements.
- Use proper formatting.
- Follow modern best practices.
- Hinglish input → Hinglish answer.
- English input → English answer.
"""
    },


    "friend": {
        "label": "😎 Friend Mode",

        "system": """
You are a chill and friendly best friend.

Rules:
- Talk casually and naturally.
- Use slang and emojis when appropriate.
- Be supportive.
- Avoid corporate language.
- Hinglish input → Hinglish slang.
- English input → casual English.
"""
    },
}


# ============================================================
# HC VERMA MODE
# ============================================================

HC_VERMA_PROMPT = """
You are an expert Physics teacher and HC Verma problem-solving assistant.

Your job is to help students understand and solve specific Physics
questions from Concepts of Physics by H.C. Verma.

IMPORTANT RULES:

1. Solve the specific question supplied by the user.
2. Do not invent a question or exercise.
3. Do not claim an answer is from HC Verma unless the user supplied
   the question or enough information to identify it.
4. Do not reproduce large portions of the textbook.
5. Focus on explanation and solving.

FOR NUMERICAL QUESTIONS:

Use this structure when appropriate:

Given:
- List all known values.

Required:
- What needs to be calculated.

Concept / Formula:
- Explain the relevant Physics principle.

Solution:
- Substitute values step by step.
- Show important calculations.

Final Answer:
- Clearly state the answer with correct SI units.

FOR CONCEPTUAL QUESTIONS:

1. Identify the Physics concept.
2. Explain the concept simply.
3. Give the reasoning step by step.
4. State the final conclusion.

FOR DERIVATIONS:

1. Start from the fundamental equation.
2. Explain each important step.
3. Clearly derive the final expression.

FOR DIAGRAM QUESTIONS:

- Carefully inspect the diagram.
- Identify relevant objects, directions, angles,
  forces, distances and labels.
- Do not assume a label that cannot be read.

If the question/image is unclear:
ask the user to send a clearer image or type the question.

Language:
- Hindi → Hindi
- Hinglish → Hinglish
- English → English

Be accurate and educational.
"""


def get_system_prompt(user_id: str) -> str:

    data = get_user_data(user_id)

    mode = data.get("mode", "default")

    prompt = AI_MODES.get(
        mode,
        AI_MODES["default"]
    )["system"]

    if data.get("hc_verma_mode", False):

        prompt += "\n\n" + HC_VERMA_PROMPT

    return prompt


def get_mode_label(user_id: str) -> str:

    data = get_user_data(user_id)

    mode = data.get("mode", "default")

    return AI_MODES.get(
        mode,
        AI_MODES["default"]
    )["label"]


# ============================================================
# TEXT HELPERS
# ============================================================

def clean_text_for_tts(text: str) -> str:

    text = re.sub(
        r"[*_`]",
        "",
        text
    )

    text = re.sub(
        r"https?://\S+",
        "",
        text
    )

    text = re.sub(
        r"[\U00010000-\U0010ffff]",
        "",
        text
    )

    text = re.sub(
        r"\s+",
        " ",
        text
    ).strip()

    return text


def detect_lang(text: str) -> str:

    hindi_chars = re.findall(
        r"[\u0900-\u097F]",
        text
    )

    if hindi_chars:
        return "hi"

    return "en"


# ============================================================
# VOICE REPLY
# ============================================================

async def send_voice_reply(
    update: Update,
    text: str,
    lang: str = "en"
):

    tmp_path = None

    try:

        clean = clean_text_for_tts(text)

        if not clean:
            return False

        clean = clean[:500]

        tts = gTTS(
            text=clean,
            lang=lang,
            slow=False
        )

        with tempfile.NamedTemporaryFile(
            suffix=".mp3",
            delete=False
        ) as tmp:

            tmp_path = tmp.name

        tts.save(tmp_path)

        with open(
            tmp_path,
            "rb"
        ) as audio:

            await update.message.reply_voice(
                voice=audio,
                caption="🔊 AI Voice Reply"
            )

        return True

    except Exception as e:

        print("TTS ERROR:", e)

        return False

    finally:

        if (
            tmp_path
            and os.path.exists(tmp_path)
        ):
            try:
                os.unlink(tmp_path)
            except Exception:
                pass


# ============================================================
# IMAGE REQUEST DETECTION
# ============================================================

IMAGE_NOUNS = [
    "image",
    "photo",
    "pic",
    "picture",
    "wallpaper",
    "logo",
    "poster",
    "illustration",
    "artwork",
    "avatar",
    "thumbnail",
    "banner",
]


IMAGE_VERBS = [
    "generate",
    "create",
    "make",
    "draw",
    "design",
    "bana",
    "bnao",
    "banado",
    "banade",
    "bnado",
]


def is_image_request(text: str) -> bool:

    t = text.lower()

    has_noun = any(
        noun in t
        for noun in IMAGE_NOUNS
    )

    if not has_noun:
        return False

    has_verb = any(
        verb in t
        for verb in IMAGE_VERBS
    )

    if has_verb:
        return True

    return any(
        noun in t
        for noun in [
            "wallpaper",
            "logo",
            "poster"
        ]
    )


# ============================================================
# /START
# ============================================================

async def start(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE
):

    user = update.message.from_user

    name = user.first_name or "there"

    user_id = str(
        update.message.chat_id
    )

    data = get_user_data(user_id)

    if data["name"] is None:

        data["name"] = name

        save_memory()

    welcome = (
        f"👋 Hey {name}! Welcome to your AI Assistant!\n\n"

        "━━━━━━━━━━━━━━━━━━━━━\n"

        "🤖 *What I can do:*\n\n"

        "💬 AI Chat — Just type anything!\n"
        "🖼️ Image Gen — Ask for an image\n"
        "👀 Vision AI — Send a photo\n"
        "🎤 Voice AI — Send voice\n"
        "📄 PDF AI — Send a PDF\n"
        "🌐 Web Search — /search anything\n"
        "🎓 AI Modes — /mode\n"
        "📚 HC Verma Solver — /hcverma\n\n"

        "━━━━━━━━━━━━━━━━━━━━━\n"

        "📜 /help\n"
        "🧹 /clear\n"
        "🎭 /mode\n"
        "📚 /hcverma\n"
        "🔊 /voicemode\n\n"

        "Let's go! 🚀🔥"
    )

    await update.message.reply_text(
        welcome,
        parse_mode="Markdown"
    )


# ============================================================
# /HELP
# ============================================================

async def help_command(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE
):

    help_text = (
        "🤖 *Available Commands:*\n\n"

        "/start — Welcome screen 🏠\n"
        "/help — Help menu 📜\n"
        "/clear — Clear memory 🧹\n"
        "/mode — AI personality 🎭\n"
        "/hcverma — HC Verma Physics 📚\n"
        "/voicemode — Voice replies 🔊\n"
        "/joke — Joke 😂\n"
        "/motivate — Motivation 🚀\n"
        "/search `<query>` — Web search 🌐\n\n"

        "━━━━━━━━━━━━━━━━━━━━━\n"

        "✨ *Features:*\n\n"

        "• 🤖 AI Chat\n"
        "• 🧠 Permanent Memory\n"
        "• 📚 HC Verma Solver\n"
        "• 📸 Physics Question Image Solver\n"
        "• 🖼️ AI Image Generation\n"
        "• 👀 Vision AI\n"
        "• 🎤 Voice AI\n"
        "• 📄 PDF Summarizer\n"
        "• 🌐 Web Search\n"
        "• 🎭 Teacher / Coder / Friend modes\n"
        "• 😄 English + Hinglish"
    )

    await update.message.reply_text(
        help_text,
        parse_mode="Markdown"
    )


# ============================================================
# /CLEAR
# ============================================================

async def clear(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE
):

    user_id = str(
        update.message.chat_id
    )

    data = get_user_data(user_id)

    data["history"] = []

    bot_messages[user_id] = []

    save_memory()

    await update.message.reply_text(
        "Memory cleared successfully 🧹✨\n"
        "Fresh start — what's on your mind?"
    )


# ============================================================
# /MODE
# ============================================================

async def mode_command(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE
):

    user_id = str(
        update.message.chat_id
    )

    if context.args:

        chosen = context.args[0].lower()

        if chosen in AI_MODES:

            data = get_user_data(user_id)

            data["mode"] = chosen

            save_memory()

            label = AI_MODES[chosen]["label"]

            await update.message.reply_text(
                f"Mode switched to {label} 🎉\n\n"
                f"I'll now talk like a {chosen}. 🔥"
            )

        else:

            modes_list = "\n".join(
                f"• `{key}` — {value['label']}"
                for key, value in AI_MODES.items()
            )

            await update.message.reply_text(
                "❌ Unknown mode.\n\n"
                f"{modes_list}\n\n"
                "Usage: `/mode teacher`",
                parse_mode="Markdown"
            )

        return

    current = get_mode_label(user_id)

    modes_list = "\n".join(
        f"• `{key}` — {value['label']}"
        for key, value in AI_MODES.items()
    )

    await update.message.reply_text(
        "🎭 *AI Modes*\n\n"
        f"Current: {current}\n\n"
        f"{modes_list}\n\n"
        "Usage: `/mode teacher`",
        parse_mode="Markdown"
    )


# ============================================================
# /H C V E R M A
# ============================================================

async def hcverma_command(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE
):

    user_id = str(
        update.message.chat_id
    )

    data = get_user_data(user_id)

    if context.args:

        choice = context.args[0].lower()

        if choice in (
            "on",
            "enable",
            "start"
        ):

            data["hc_verma_mode"] = True

        elif choice in (
            "off",
            "disable",
            "stop"
        ):

            data["hc_verma_mode"] = False

        else:

            await update.message.reply_text(
                "Usage:\n"
                "/hcverma on\n"
                "/hcverma off"
            )

            return

    else:

        data["hc_verma_mode"] = not data.get(
            "hc_verma_mode",
            False
        )

    save_memory()

    if data["hc_verma_mode"]:

        await update.message.reply_text(
            "📚 *HC Verma Mode ON!*\n\n"

            "HC Verma ka Physics question "
            "text mein bhejo ya question ki "
            "photo bhejo. 📸\n\n"

            "🧮 Numerical:\n"
            "Given → Required → Formula → "
            "Calculation → Final Answer\n\n"

            "🎓 Conceptual:\n"
            "Concept → Reasoning → Answer\n\n"

            "Band karne ke liye:\n"
            "`/hcverma off`",
            parse_mode="Markdown"
        )

    else:

        await update.message.reply_text(
            "📚 HC Verma Mode OFF.\n\n"
            "Back to normal AI mode 🤖"
        )


# ============================================================
# /VOICE MODE
# ============================================================

async def voicemode_command(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE
):

    user_id = str(
        update.message.chat_id
    )

    data = get_user_data(user_id)

    data["voice_mode"] = not data.get(
        "voice_mode",
        False
    )

    save_memory()

    if data["voice_mode"]:

        await update.message.reply_text(
            "🔊 *Voice Mode ON!*\n\n"
            "I'll reply with text + voice 🎤✨\n\n"
            "Type `/voicemode` again to turn off.",
            parse_mode="Markdown"
        )

    else:

        await update.message.reply_text(
            "🔇 *Voice Mode OFF.*\n\n"
            "Text-only replies now.",
            parse_mode="Markdown"
        )


# ============================================================
# /JOKE
# ============================================================

async def joke(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE
):

    jokes = [
        "Why don't programmers like nature? 🌳 Because it has too many bugs 😂",
        "Coding at 2AM hits different 😵‍💫☕",
        "My brain has too many tabs open 🤯😂",
        "Why do Java developers wear glasses? 👓 Because they don't C# 😂",
        "404: Sleep not found 💀😂",
    ]

    await update.message.reply_text(
        random.choice(jokes)
    )


# ============================================================
# /MOTIVATE
# ============================================================

async def motivate(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE
):

    motivation = [
        "Consistency beats motivation 💪🔥",
        "Small progress is still progress 🚀",
        "Don't stop until you're proud 😎🔥",
        "You're one decision away from a different life 🌟",
        "The grind never lies 💯🔥",
    ]

    await update.message.reply_text(
        random.choice(motivation)
    )


# ============================================================
# /SEARCH
# ============================================================

async def search(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE
):

    query = " ".join(
        context.args
    ).strip()

    if not query:

        await update.message.reply_text(
            "Usage:\n"
            "/search artificial intelligence"
        )

        return

    await context.bot.send_chat_action(
        chat_id=update.message.chat_id,
        action=ChatAction.TYPING
    )

    try:

        url = (
            "https://api.duckduckgo.com/"
            f"?q={urllib.parse.quote(query)}"
            "&format=json"
        )

        response = requests.get(
            url,
            timeout=30,
            headers={
                "User-Agent": "Mozilla/5.0"
            }
        )

        response.raise_for_status()

        result = response.json().get(
            "AbstractText"
        )

        if result:

            await update.message.reply_text(
                "🔎 *Search Result:*\n\n"
                + result[:3500],
                parse_mode="Markdown"
            )

        else:

            completion = client.chat.completions.create(
                model=CHAT_MODEL,

                messages=[
                    {
                        "role": "system",
                        "content": (
                            "Answer naturally and briefly. "
                            "The user asked for information "
                            "about the following search query."
                        ),
                    },
                    {
                        "role": "user",
                        "content": query,
                    },
                ],
            )

            ai_result = (
                completion
                .choices[0]
                .message
                .content
            )

            await update.message.reply_text(
                "🤖 *AI Answer:*\n\n"
                + ai_result[:3500],
                parse_mode="Markdown"
            )

    except Exception as e:

        print("SEARCH ERROR:", e)

        await update.message.reply_text(
            "❌ Search failed. Try again!"
        )


# ============================================================
# AI IMAGE GENERATION
# ============================================================

async def generate_ai_image(
    update: Update,
    prompt: str
):

    try:

        await update.message.reply_text(
            "🎨 Creating your AI image... ✨"
        )

        enhance_prompt = f"""
Convert the following user request into
a professional AI image generation prompt.

User Request:
{prompt}

Make the image:
- cinematic
- realistic
- highly detailed
- visually attractive

Return ONLY the final image prompt.
"""

        completion = client.chat.completions.create(

            model=CHAT_MODEL,

            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a professional "
                        "AI image prompt engineer."
                    ),
                },
                {
                    "role": "user",
                    "content": enhance_prompt,
                },
            ],
        )

        final_prompt = (
            completion
            .choices[0]
            .message
            .content
            .strip()
        )

        encoded_prompt = urllib.parse.quote(
            final_prompt
        )

        image_url = (
            "https://image.pollinations.ai/prompt/"
            + encoded_prompt
        )

        response = requests.get(
            image_url,
            timeout=60,
            headers={
                "User-Agent": "Mozilla/5.0"
            }
        )

        if response.status_code == 200:

            await update.message.reply_photo(
                photo=response.content,
                caption="🖼️ Here's your AI image 😄🔥"
            )

        else:

            await update.message.reply_text(
                "❌ Image server error. Try again!"
            )

    except Exception as e:

        print("IMAGE ERROR:", e)

        await update.message.reply_text(
            "❌ Image generation failed. Try again!"
        )


# ============================================================
# PHOTO / VISION AI
# ============================================================

async def handle_photo(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE
):

    try:

        await update.message.reply_text(
            "👀 Analyzing image... 🔍✨"
        )

        photo = update.message.photo[-1]

        telegram_file = await context.bot.get_file(
            photo.file_id
        )

        photo_bytes = (
            await telegram_file.download_as_bytearray()
        )

        base64_image = base64.b64encode(
            bytes(photo_bytes)
        ).decode("utf-8")

        user_id = str(
            update.message.chat_id
        )

        data = get_user_data(user_id)

        if data.get(
            "hc_verma_mode",
            False
        ):

            user_text = (
                update.message.caption
                or
                """
Read the Physics question in this image.
Solve it completely step by step.

If it is numerical:
Given → Required → Formula → Calculation → Final Answer.

If it is conceptual:
Concept → Reasoning → Final Answer.

Carefully inspect any diagram.
"""
            )

            user_text += (
                "\n\n"
                + HC_VERMA_PROMPT
            )

        else:

            user_text = (
                update.message.caption
                or
                "Describe this image briefly. "
                "Mention important objects and actions."
            )

        completion = client.chat.completions.create(

            model=VISION_MODEL,

            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": user_text,
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url":
                                "data:image/jpeg;base64,"
                                + base64_image
                            },
                        },
                    ],
                }
            ],
        )

        ai_reply = (
            completion
            .choices[0]
            .message
            .content
        )

        await update.message.reply_text(
            "👁️ *Vision AI:*\n\n"
            + ai_reply[:4000],
            parse_mode="Markdown"
        )

    except Exception as e:

        print("VISION ERROR:", e)

        await update.message.reply_text(
            "❌ Image analysis failed.\n"
            "Try sending the image again."
        )


# ============================================================
# VOICE AI
# ============================================================

async def handle_voice(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE
):

    tmp_path = None

    try:

        await update.message.reply_text(
            "🎤 Processing your voice message... 🔊✨"
        )

        voice = update.message.voice

        telegram_file = await context.bot.get_file(
            voice.file_id
        )

        with tempfile.NamedTemporaryFile(
            suffix=".ogg",
            delete=False
        ) as tmp:

            tmp_path = tmp.name

        await telegram_file.download_to_drive(
            tmp_path
        )

        with open(
            tmp_path,
            "rb"
        ) as audio_file:

            transcription = client.audio.transcriptions.create(

                model=WHISPER_MODEL,

                file=audio_file,

                response_format="text"
            )

        if isinstance(
            transcription,
            str
        ):

            transcript_text = transcription.strip()

        else:

            transcript_text = str(
                transcription
            ).strip()

        if not transcript_text:

            await update.message.reply_text(
                "❌ Couldn't understand the audio."
            )

            return

        await update.message.reply_text(
            "🎙️ *You said:*\n"
            + transcript_text,
            parse_mode="Markdown"
        )

        user_id = str(
            update.message.chat_id
        )

        data = get_user_data(
            user_id
        )

        data["history"].append(
            {
                "role": "user",
                "content": transcript_text,
            }
        )

        data["history"] = data[
            "history"
        ][-30:]

        save_memory()

        messages = [
            {
                "role": "system",
                "content": get_system_prompt(
                    user_id
                ),
            }
        ] + data["history"][-20:]

        completion = client.chat.completions.create(

            model=CHAT_MODEL,

            messages=messages
        )

        ai_reply = (
            completion
            .choices[0]
            .message
            .content
        )

        data["history"].append(
            {
                "role": "assistant",
                "content": ai_reply,
            }
        )

        data["history"] = data[
            "history"
        ][-30:]

        save_memory()

        await update.message.reply_text(
            "🤖 *AI:*\n\n"
            + ai_reply[:3500],
            parse_mode="Markdown"
        )

        await context.bot.send_chat_action(
            chat_id=update.message.chat_id,
            action=ChatAction.RECORD_VOICE
        )

        await send_voice_reply(
            update,
            ai_reply,
            lang=detect_lang(
                transcript_text
            )
        )

    except Exception as e:

        print("VOICE ERROR:", e)

        await update.message.reply_text(
            "❌ Voice processing failed. "
            "Try again!"
        )

    finally:

        if (
            tmp_path
            and os.path.exists(tmp_path)
        ):

            try:
                os.unlink(tmp_path)
            except Exception:
                pass


# ============================================================
# PDF AI
# ============================================================

async def handle_document(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE
):

    doc = update.message.document

    if not doc:
        return

    if not doc.file_name:

        await update.message.reply_text(
            "❌ PDF filename missing."
        )

        return

    if not doc.file_name.lower().endswith(
        ".pdf"
    ):

        await update.message.reply_text(
            "📄 Please send a PDF file."
        )

        return

    tmp_path = None

    try:

        await update.message.reply_text(
            "📄 Reading your PDF... 🔍✨"
        )

        telegram_file = await context.bot.get_file(
            doc.file_id
        )

        with tempfile.NamedTemporaryFile(
            suffix=".pdf",
            delete=False
        ) as tmp:

            tmp_path = tmp.name

        await telegram_file.download_to_drive(
            tmp_path
        )

        try:

            from pypdf import PdfReader

        except ImportError:

            await update.message.reply_text(
                "❌ pypdf is not installed.\n\n"
                "Run:\n"
                "pip install pypdf"
            )

            return

        reader = PdfReader(
            tmp_path
        )

        text_parts = []

        for page in reader.pages:

            page_text = page.extract_text()

            if page_text:
                text_parts.append(
                    page_text
                )

        text = "\n".join(
            text_parts
        )

        if not text.strip():

            await update.message.reply_text(
                "❌ Couldn't extract text from this PDF."
            )

            return

        text_excerpt = text[:12000]

        caption = (
            update.message.caption
            or
            "Summarize this PDF clearly."
        )

        completion = client.chat.completions.create(

            model=CHAT_MODEL,

            messages=[
                {
                    "role": "system",
                    "content": """
You are a PDF summarizer.

Extract:
- Main ideas
- Important points
- Key details
- Important formulas if present

Use clear bullet points.
""",
                },
                {
                    "role": "user",
                    "content":
                    caption
                    + "\n\nPDF Content:\n"
                    + text_excerpt,
                },
            ],
        )

        ai_reply = (
            completion
            .choices[0]
            .message
            .content
        )

        await update.message.reply_text(
            "📄 *PDF Summary:*\n\n"
            + ai_reply[:4000],
            parse_mode="Markdown"
        )

    except Exception as e:

        print("PDF ERROR:", e)

        await update.message.reply_text(
            "❌ PDF processing failed."
        )

    finally:

        if (
            tmp_path
            and os.path.exists(tmp_path)
        ):

            try:
                os.unlink(tmp_path)
            except Exception:
                pass


# ============================================================
# MAIN CHAT
# ============================================================

async def reply(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE
):

    if not update.message:
        return

    user_message = update.message.text

    if not user_message:
        return

    user_message = user_message.strip()

    if not user_message:
        return

    user_id = str(
        update.message.chat_id
    )

    await context.bot.send_chat_action(
        chat_id=update.message.chat_id,
        action=ChatAction.TYPING
    )

    data = get_user_data(
        user_id
    )

    # --------------------------------------------------------
    # CLEAR CHAT TEXT
    # --------------------------------------------------------

    if user_message.lower() in [
        "clear chat",
        "clear memory",
    ]:

        data["history"] = []

        bot_messages[user_id] = []

        save_memory()

        await update.message.reply_text(
            "Memory cleared successfully 🧹✨"
        )

        return

    # --------------------------------------------------------
    # IMAGE REQUEST
    # --------------------------------------------------------

    if is_image_request(
        user_message
    ):

        await generate_ai_image(
            update,
            user_message
        )

        return

    # --------------------------------------------------------
    # SAVE USER MESSAGE
    # --------------------------------------------------------

    data["history"].append(
        {
            "role": "user",
            "content": user_message,
        }
    )

    data["history"] = data[
        "history"
    ][-30:]

    save_memory()

    # --------------------------------------------------------
    # BUILD PROMPT
    # --------------------------------------------------------

    system_prompt = get_system_prompt(
        user_id
    )

    messages = [
        {
            "role": "system",
            "content": system_prompt,
        }
    ] + data["history"][-20:]

    # --------------------------------------------------------
    # AI
    # --------------------------------------------------------

    try:

        completion = client.chat.completions.create(

            model=CHAT_MODEL,

            messages=messages,

            temperature=0.7,

            max_tokens=2000
        )

        ai_reply = (
            completion
            .choices[0]
            .message
            .content
        )

        if not ai_reply:
            ai_reply = "I couldn't generate a reply."

    except Exception as e:

        print("AI ERROR:", e)

        ai_reply = (
            "Oops! Something went wrong on my end 😵\n"
            "Please try again."
        )

    # --------------------------------------------------------
    # SAVE AI RESPONSE
    # --------------------------------------------------------

    data["history"].append(
        {
            "role": "assistant",
            "content": ai_reply,
        }
    )

    data["history"] = data[
        "history"
    ][-30:]

    save_memory()

    # --------------------------------------------------------
    # SEND RESPONSE
    # --------------------------------------------------------

    try:

        sent_msg = await update.message.reply_text(
            str(ai_reply)[:4000]
        )

        if user_id not in bot_messages:
            bot_messages[user_id] = []

        bot_messages[user_id].append(
            sent_msg.message_id
        )

        # Voice Mode
        if data.get(
            "voice_mode",
            False
        ):

            await context.bot.send_chat_action(
                chat_id=update.message.chat_id,
                action=ChatAction.RECORD_VOICE
            )

            await send_voice_reply(
                update,
                ai_reply,
                lang=detect_lang(
                    ai_reply
                )
            )

    except Exception as e:

        print("SEND ERROR:", e)


# ============================================================
# ERROR HANDLER
# ============================================================

async def error_handler(
    update: object,
    context: ContextTypes.DEFAULT_TYPE
):

    print(
        "UNHANDLED ERROR:",
        context.error
    )

    try:

        if isinstance(
            update,
            Update
        ) and update.message:

            await update.message.reply_text(
                "❌ Something went wrong. "
                "Please try again."
            )

    except Exception as e:

        print(
            "ERROR HANDLER ERROR:",
            e
        )


# ============================================================
# RENDER HEALTH SERVER
# ============================================================

import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


class HealthHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.send_header("Content-Type", "text/plain; charset=utf-8")
        self.end_headers()
        self.wfile.write(b"Stack AI Bot is running")

    def do_HEAD(self):
        self.send_response(200)
        self.send_header("Content-Type", "text/plain; charset=utf-8")
        self.end_headers()

    def log_message(self, format, *args):
        return


def start_health_server():
    port = int(os.getenv("PORT", "10000"))
    server = ThreadingHTTPServer(("0.0.0.0", port), HealthHandler)
    print(f"Health server listening on 0.0.0.0:{port}")
    server.serve_forever()


# ============================================================
# MAIN
# ============================================================

def main():

    print("================================")
    print("🤖 Starting Telegram AI Bot...")
    print("================================")

    app = (
        ApplicationBuilder()
        .token(BOT_TOKEN)
        .connect_timeout(60)
        .read_timeout(60)
        .write_timeout(60)
        .pool_timeout(60)
        .build()
    )

    # --------------------------------------------------------
    # COMMANDS
    # --------------------------------------------------------

    app.add_handler(
        CommandHandler(
            "start",
            start
        )
    )

    app.add_handler(
        CommandHandler(
            "help",
            help_command
        )
    )

    app.add_handler(
        CommandHandler(
            "clear",
            clear
        )
    )

    app.add_handler(
        CommandHandler(
            "mode",
            mode_command
        )
    )

    app.add_handler(
        CommandHandler(
            "hcverma",
            hcverma_command
        )
    )

    app.add_handler(
        CommandHandler(
            "voicemode",
            voicemode_command
        )
    )

    app.add_handler(
        CommandHandler(
            "joke",
            joke
        )
    )

    app.add_handler(
        CommandHandler(
            "motivate",
            motivate
        )
    )

    app.add_handler(
        CommandHandler(
            "search",
            search
        )
    )

    # --------------------------------------------------------
    # MEDIA HANDLERS
    # --------------------------------------------------------

    app.add_handler(
        MessageHandler(
            filters.PHOTO & ~filters.COMMAND,
            handle_photo
        )
    )

    app.add_handler(
        MessageHandler(
            filters.VOICE & ~filters.COMMAND,
            handle_voice
        )
    )

    app.add_handler(
        MessageHandler(
            filters.Document.ALL & ~filters.COMMAND,
            handle_document
        )
    )

    # --------------------------------------------------------
    # NORMAL TEXT
    # --------------------------------------------------------

    app.add_handler(
        MessageHandler(
            filters.TEXT & ~filters.COMMAND,
            reply
        )
    )

    # --------------------------------------------------------
    # ERROR HANDLER
    # --------------------------------------------------------

    app.add_error_handler(
        error_handler
    )

    print("🤖 AI Bot is running...")
    print("📚 HC Verma Mode available:")
    print("   /hcverma on")
    print("   /hcverma off")
    print("================================")

    # Render Web Services require an HTTP port to be open.
    # The Telegram bot continues to use long polling in the main thread.
    threading.Thread(target=start_health_server, daemon=True).start()

    app.run_polling(
        drop_pending_updates=True,
        allowed_updates=Update.ALL_TYPES,
        timeout=30,
        bootstrap_retries=10
    )


# ============================================================
# START
# ============================================================

if __name__ == "__main__":
    main()