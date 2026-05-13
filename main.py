from telegram import Update
from telegram.ext import (
    ApplicationBuilder,
    MessageHandler,
    CommandHandler,
    filters,
    ContextTypes
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
import asyncio

# Load env
load_dotenv()

# Tokens
BOT_TOKEN = os.getenv("BOT_TOKEN")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# Load memory
try:
    with open("memory.json", "r") as file:
        user_memory = json.load(file)
except Exception:
    user_memory = {}

# Bot messages tracker
bot_messages = {}

# Groq client
client = Groq(api_key=GROQ_API_KEY)

# ─── AI MODES ────────────────────────────────────────────────────────────────

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
- Never write essays.
- Avoid robotic AI language.
- Sound friendly and modern.
- Use emojis naturally 😄🔥.
- If user talks casually, reply casually.
- If user talks in Hinglish, reply in Hinglish.
- If user talks in English, reply in English.
"""
    },
    "teacher": {
        "label": "🎓 Teacher Mode",
        "system": """
You are an expert teacher and educator.
- Explain concepts clearly with examples and analogies.
- Break down complex topics step by step.
- Encourage the user and make learning fun 📚✨.
- Use simple language unless technical depth is requested.
- Add helpful tips and memory tricks when relevant.
- If user talks in Hinglish, reply in Hinglish; English → English.
"""
    },
    "coder": {
        "label": "💻 Coder Mode",
        "system": """
You are an expert software engineer and coding assistant.
- Always provide clean, well-commented code.
- Explain what the code does after each snippet.
- Point out potential bugs or improvements.
- Use modern best practices.
- Format code with proper indentation.
- If user talks in Hinglish, reply in Hinglish; English → English.
"""
    },
    "friend": {
        "label": "😎 Friend Mode",
        "system": """
You are a chill, fun best friend.
- Talk casually and naturally.
- Use slang, emojis, and humor freely 😂🔥.
- Be supportive and understanding.
- Keep it real — no corporate speak.
- If user talks in Hinglish, reply in Hinglish slang; English → casual English.
"""
    }
}

# ─── HELPERS ─────────────────────────────────────────────────────────────────

def save_memory():
    with open("memory.json", "w") as file:
        json.dump(user_memory, file)


def clean_text_for_tts(text: str) -> str:
    """Remove emojis, markdown, and special chars for clean TTS."""
    # Remove markdown bold/italic
    text = re.sub(r"[*_`]", "", text)
    # Remove emojis
    text = re.sub(
        r"[\U00010000-\U0010ffff"
        r"\U0001F600-\U0001F64F"
        r"\U0001F300-\U0001F5FF"
        r"\U0001F680-\U0001F6FF"
        r"\U0001F1E0-\U0001F1FF"
        r"\u2600-\u26FF\u2700-\u27BF]+",
        "",
        text,
        flags=re.UNICODE
    )
    # Remove URLs
    text = re.sub(r"https?://\S+", "", text)
    # Clean up extra whitespace
    text = re.sub(r"\s+", " ", text).strip()
    return text


async def send_voice_reply(update: Update, text: str, lang: str = "en"):
    """Convert AI text reply to voice and send as Telegram voice note."""
    try:
        clean = clean_text_for_tts(text)
        if not clean:
            return False

        # Limit TTS length (very long text is slow)
        clean = clean[:500]

        tts = gTTS(text=clean, lang=lang, slow=False)

        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp:
            tmp_path = tmp.name

        tts.save(tmp_path)

        with open(tmp_path, "rb") as audio:
            await update.message.reply_voice(
                voice=audio,
                caption="🔊 AI Voice Reply"
            )

        os.unlink(tmp_path)
        return True

    except Exception as e:
        print(f"TTS ERROR: {e}")
        return False


def detect_lang(text: str) -> str:
    """Detect if text is Hindi/Hinglish or English for TTS lang."""
    # Simple heuristic: if Hindi unicode chars present, use Hindi
    hindi_chars = re.findall(r"[\u0900-\u097F]", text)
    if hindi_chars:
        return "hi"
    return "en"


def get_user_data(user_id: str) -> dict:
    """Get or initialize user data."""
    if user_id not in user_memory:
        user_memory[user_id] = {
            "history": [],
            "mode": "default",
            "name": None
        }
    # Migrate old flat list format
    if isinstance(user_memory[user_id], list):
        user_memory[user_id] = {
            "history": user_memory[user_id],
            "mode": "default",
            "name": None
        }
    return user_memory[user_id]


def get_system_prompt(user_id: str) -> str:
    data = get_user_data(user_id)
    mode = data.get("mode", "default")
    return AI_MODES.get(mode, AI_MODES["default"])["system"]


def get_mode_label(user_id: str) -> str:
    data = get_user_data(user_id)
    mode = data.get("mode", "default")
    return AI_MODES.get(mode, AI_MODES["default"])["label"]


# ─── COMMANDS ─────────────────────────────────────────────────────────────────

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user = update.message.from_user
    name = user.first_name or "there"
    user_id = str(update.message.chat_id)

    # Initialize user
    data = get_user_data(user_id)
    if data["name"] is None:
        data["name"] = name
        save_memory()

    welcome = (
        f"👋 Hey {name}! Welcome to your AI Assistant!\n\n"
        "━━━━━━━━━━━━━━━━━━━━━\n"
        "🤖 *What I can do:*\n\n"
        "💬 AI Chat — Just type anything!\n"
        "🖼️ Image Gen — 'make a wallpaper of sunset'\n"
        "👀 Vision AI — Send any photo!\n"
        "🎤 Voice AI — Send a voice message!\n"
        "📄 PDF AI — Send a PDF for summary!\n"
        "🌐 Web Search — /search anything\n"
        "🎓 AI Modes — /mode to switch persona\n\n"
        "━━━━━━━━━━━━━━━━━━━━━\n"
        "📜 /help — All commands\n"
        "🧹 /clear — Reset memory\n"
        "🎭 /mode — Switch AI personality\n\n"
        "Let's go! Ask me anything 🚀🔥"
    )

    await update.message.reply_text(welcome, parse_mode="Markdown")


async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    help_text = (
        "🤖 *Available Commands:*\n\n"
        "/start — Welcome screen 🏠\n"
        "/help — Show this menu 📜\n"
        "/clear — Clear memory 🧹\n"
        "/mode — Switch AI mode 🎭\n"
        "/voicemode — Toggle voice replies 🔊\n"
        "/joke — Funny joke 😂\n"
        "/motivate — Daily motivation 🚀\n"
        "/search `<query>` — Search internet 🌐\n\n"
        "━━━━━━━━━━━━━━━━━━━━━\n"
        "✨ *Smart Features:*\n\n"
        "• 🤖 AI Chat (memory enabled)\n"
        "• 🖼️ Auto AI Image Generation\n"
        "• 👀 Vision AI (send photos)\n"
        "• 🎤 Voice AI (send voice notes)\n"
        "• 📄 PDF Summarizer (send PDFs)\n"
        "• 🌐 Live Web Search\n"
        "• 🎭 AI Modes: Teacher / Coder / Friend\n"
        "• 🧠 Permanent Memory\n"
        "• 😄 English + Hinglish"
    )
    await update.message.reply_text(help_text, parse_mode="Markdown")


async def clear(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = str(update.message.chat_id)
    data = get_user_data(user_id)
    data["history"] = []
    bot_messages[user_id] = []
    save_memory()
    await update.message.reply_text("Memory cleared successfully 🧹✨\nFresh start — what's on your mind?")


async def mode_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = str(update.message.chat_id)

    if context.args:
        chosen = context.args[0].lower()
        if chosen in AI_MODES:
            data = get_user_data(user_id)
            data["mode"] = chosen
            save_memory()
            label = AI_MODES[chosen]["label"]
            await update.message.reply_text(
                f"Mode switched to {label} 🎉\n\nI'll now talk like a {chosen}. Let's go! 🔥"
            )
        else:
            modes_list = "\n".join(
                [f"• `{k}` — {v['label']}" for k, v in AI_MODES.items()]
            )
            await update.message.reply_text(
                f"❌ Unknown mode. Available modes:\n\n{modes_list}\n\n"
                "Usage: `/mode teacher`",
                parse_mode="Markdown"
            )
        return

    # Show mode menu
    current = get_mode_label(user_id)
    modes_list = "\n".join(
        [f"• `{k}` — {v['label']}" for k, v in AI_MODES.items()]
    )
    await update.message.reply_text(
        f"🎭 *AI Modes*\n\n"
        f"Current: {current}\n\n"
        f"{modes_list}\n\n"
        "Usage: `/mode teacher` or `/mode coder`",
        parse_mode="Markdown"
    )


async def voicemode_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Toggle voice replies for text chat."""
    user_id = str(update.message.chat_id)
    data = get_user_data(user_id)
    current = data.get("voice_mode", False)
    data["voice_mode"] = not current
    save_memory()

    if data["voice_mode"]:
        await update.message.reply_text(
            "🔊 *Voice Mode ON!*\n\nI'll reply with voice + text now 🎤✨\n\nType `/voicemode` again to turn off.",
            parse_mode="Markdown"
        )
    else:
        await update.message.reply_text(
            "🔇 *Voice Mode OFF.*\n\nText-only replies now. Type `/voicemode` to turn back on.",
            parse_mode="Markdown"
        )


async def joke(update: Update, context: ContextTypes.DEFAULT_TYPE):
    jokes = [
        "Why don't programmers like nature? 🌳 Because it has too many bugs 😂",
        "Coding at 2AM hits different 😵‍💫☕",
        "My brain has too many tabs open 🤯😂",
        "Why do Java developers wear glasses? 👓 Because they don't C# 😂",
        "404: Sleep not found 💀😂"
    ]
    await update.message.reply_text(random.choice(jokes))


async def motivate(update: Update, context: ContextTypes.DEFAULT_TYPE):
    motivation = [
        "Consistency beats motivation 💪🔥",
        "Small progress is still progress 🚀😄",
        "Don't stop until you're proud 😎🔥",
        "You're one decision away from a different life 🌟",
        "The grind never lies 💯🔥"
    ]
    await update.message.reply_text(random.choice(motivation))


async def search(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = " ".join(context.args)

    if not query:
        await update.message.reply_text("Usage: /search artificial intelligence 🔍")
        return

    await context.bot.send_chat_action(
        chat_id=update.message.chat_id,
        action=ChatAction.TYPING
    )

    try:
        url = f"https://api.duckduckgo.com/?q={urllib.parse.quote(query)}&format=json"
        response = requests.get(url, timeout=30).json()
        result = response.get("AbstractText")

        if result:
            await update.message.reply_text(f"🔎 *Search Result:*\n\n{result}", parse_mode="Markdown")
        else:
            # Fallback: ask AI with web context
            await update.message.reply_text(
                f"🌐 No direct snippet found. Let me ask AI about: *{query}*",
                parse_mode="Markdown"
            )
            # Use AI as fallback
            chat_completion = client.chat.completions.create(
                messages=[
                    {
                        "role": "system",
                        "content": "Reply naturally and shortly like ChatGPT. Keep answers under 5 lines. Never give code unless user asks for programming help."
                    },
                    {
                        "role": "user",
                        "content": f"Search query: {query}"
                    }
                ],
                model="llama-3.3-70b-versatile",
            )
            ai_result = chat_completion.choices[0].message.content
            await update.message.reply_text(f"🤖 *AI Answer:*\n\n{ai_result[:3000]}", parse_mode="Markdown")

    except Exception as e:
        print(f"SEARCH ERROR: {e}")
        await update.message.reply_text("Search failed 😵 Try again!")


# ─── IMAGE GENERATION ────────────────────────────────────────────────────────

async def generate_ai_image(update: Update, prompt: str):
    try:
        await update.message.reply_text("🎨 Creating your AI image... hang on! ✨")

        enhance_prompt = f"""
Convert this user request into a professional AI image prompt.

User Request:
{prompt}

Make it:
- cinematic and realistic
- highly detailed and visually stunning
- suitable for AI image generation

Only return the final image prompt. Nothing else.
"""

        chat_completion = client.chat.completions.create(
            messages=[
                {"role": "system", "content": "You are a professional AI image prompt engineer."},
                {"role": "user", "content": enhance_prompt}
            ],
            model="llama-3.3-70b-versatile",
        )

        final_prompt = chat_completion.choices[0].message.content.strip()
        print(f"IMAGE PROMPT: {final_prompt}")

        encoded_prompt = urllib.parse.quote(final_prompt)
        image_url = f"https://image.pollinations.ai/prompt/{encoded_prompt}"

        response = requests.get(
            image_url,
            timeout=60,
            headers={"User-Agent": "Mozilla/5.0"}
        )

        if response.status_code == 200:
            await update.message.reply_photo(
                photo=response.content,
                caption="🖼️ Here's your AI image 😄🔥",
                parse_mode="Markdown"
            )
        else:
            await update.message.reply_text("❌ Image server error. Try again! 😵")

    except Exception as e:
        print(f"IMAGE ERROR: {e}")
        await update.message.reply_text("❌ Image generation failed. Try again! 😵")


# ─── VISION AI ───────────────────────────────────────────────────────────────

async def handle_photo(update: Update, context: ContextTypes.DEFAULT_TYPE):
    try:
        await update.message.reply_text("👀 Analyzing image... 🔍✨")

        photo = update.message.photo[-1]
        file = await context.bot.get_file(photo.file_id)
        photo_bytes = await file.download_as_bytearray()
        base64_image = base64.b64encode(photo_bytes).decode("utf-8")

        user_text = (
    update.message.caption
    or "Describe this image briefly like a human. Mention important objects and actions only."
)

        chat_completion = client.chat.completions.create(
            model="meta-llama/llama-4-scout-17b-16e-instruct",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": user_text},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}
                        }
                    ]
                }
            ]
        )

        ai_reply = chat_completion.choices[0].message.content
        await update.message.reply_text(f"👁️ *Vision AI:*\n\n{ai_reply[:3000]}", parse_mode="Markdown")

    except Exception as e:
        print(f"VISION ERROR: {e}")
        await update.message.reply_text("❌ Image analysis failed. Try again! 😵")


# ─── VOICE AI ────────────────────────────────────────────────────────────────

async def handle_voice(update: Update, context: ContextTypes.DEFAULT_TYPE):
    try:
        await update.message.reply_text("🎤 Processing your voice message... 🔊✨")

        voice = update.message.voice
        file = await context.bot.get_file(voice.file_id)

        # Download voice file to temp
        with tempfile.NamedTemporaryFile(suffix=".ogg", delete=False) as tmp:
            tmp_path = tmp.name

        await file.download_to_drive(tmp_path)

        # Transcribe with Groq Whisper
        with open(tmp_path, "rb") as audio_file:
            transcription = client.audio.transcriptions.create(
                model="whisper-large-v3",
                file=audio_file,
                response_format="text"
            )

        os.unlink(tmp_path)  # Clean up temp file

        transcript_text = transcription.strip() if isinstance(transcription, str) else str(transcription)

        if not transcript_text:
            await update.message.reply_text("❌ Couldn't understand the audio. Try again! 😵")
            return

        await update.message.reply_text(f"🎙️ *You said:*\n_{transcript_text}_", parse_mode="Markdown")

        # Now process the transcribed text through AI
        user_id = str(update.message.chat_id)
        await context.bot.send_chat_action(chat_id=user_id, action=ChatAction.TYPING)

        data = get_user_data(user_id)
        data["history"].append({"role": "user", "content": transcript_text})
        save_memory()

        messages = [{"role": "system", "content": get_system_prompt(user_id)}] + data["history"][-20:]

        chat_completion = client.chat.completions.create(
            messages=messages,
            model="llama-3.3-70b-versatile",
        )

        ai_reply = chat_completion.choices[0].message.content
        data["history"].append({"role": "assistant", "content": ai_reply})
        save_memory()

        # Send text reply
        await update.message.reply_text(f"🤖 *AI:*\n\n{ai_reply[:3000]}", parse_mode="Markdown")

        # Send voice reply 🔊
        await context.bot.send_chat_action(chat_id=user_id, action=ChatAction.RECORD_VOICE)
        lang = detect_lang(transcript_text)
        await send_voice_reply(update, ai_reply, lang=lang)

    except Exception as e:
        print(f"VOICE ERROR: {e}")
        await update.message.reply_text("❌ Voice processing failed. Try again! 🎤😵")


# ─── PDF AI ──────────────────────────────────────────────────────────────────

async def handle_document(update: Update, context: ContextTypes.DEFAULT_TYPE):
    doc = update.message.document

    if not doc or not doc.file_name:
        return

    if not doc.file_name.lower().endswith(".pdf"):
        await update.message.reply_text("📄 Send a PDF file for AI summary! Only PDFs supported right now.")
        return

    try:
        await update.message.reply_text("📄 Reading your PDF... 🔍✨ This may take a moment!")

        file = await context.bot.get_file(doc.file_id)

        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
            tmp_path = tmp.name

        await file.download_to_drive(tmp_path)

        # Extract text from PDF using pypdf
        try:
            from pypdf import PdfReader
            reader = PdfReader(tmp_path)
            text = ""
            for page in reader.pages:
                text += page.extract_text() or ""
        except ImportError:
            os.unlink(tmp_path)
            await update.message.reply_text(
                "❌ PDF library not installed.\nRun: `pip install pypdf`",
                parse_mode="Markdown"
            )
            return

        os.unlink(tmp_path)

        if not text.strip():
            await update.message.reply_text("❌ Couldn't extract text from this PDF. It may be scanned/image-based.")
            return

        # Limit text for API
        text_excerpt = text[:8000]

        user_caption = update.message.caption or "Summarize this PDF clearly."

        chat_completion = client.chat.completions.create(
            messages=[
                {
                    "role": "system",
                    "content": "You are a PDF summarizer. Extract key points, main ideas, and important details from the document. Be clear, organized, and concise. Use bullet points where helpful."
                },
                {
                    "role": "user",
                    "content": f"{user_caption}\n\nPDF Content:\n{text_excerpt}"
                }
            ],
            model="llama-3.3-70b-versatile",
        )

        ai_reply = chat_completion.choices[0].message.content
        await update.message.reply_text(
            f"📄 *PDF Summary:*\n\n{ai_reply[:3500]}",
            parse_mode="Markdown"
        )

    except Exception as e:
        print(f"PDF ERROR: {e}")
        await update.message.reply_text("❌ PDF processing failed. Try again! 📄😵")


# ─── MAIN CHAT ────────────────────────────────────────────────────────────────

async def reply(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = str(update.message.chat_id)
    user_message = update.message.text

    await context.bot.send_chat_action(chat_id=user_id, action=ChatAction.TYPING)

    data = get_user_data(user_id)

    # Clear chat shortcut
    if user_message.lower() in ["clear chat", "/clear chat"]:
        data["history"] = []
        bot_messages[user_id] = []
        save_memory()
        await update.message.reply_text("Memory cleared successfully 🧹✨")
        return

    # Automatic Image Detection
    image_keywords = [
        "image", "photo", "pic", "picture", "draw", "generate", "create",
        "bnao", "banado", "anime", "wallpaper", "logo", "poster",
        "girl", "boy", "car", "bike", "illustration", "artwork", "render"
    ]

    if any(word in user_message.lower() for word in image_keywords):
        await generate_ai_image(update, user_message)
        return

    # Save user message (keep last 30 messages for memory)
    data["history"].append({"role": "user", "content": user_message})
    if len(data["history"]) > 30:
        data["history"] = data["history"][-30:]
    save_memory()

    # Build messages
    messages = [{"role": "system", "content": get_system_prompt(user_id)}] + data["history"]

    try:
        chat_completion = client.chat.completions.create(
            messages=messages,
            model="llama-3.3-70b-versatile",
        )
        ai_reply = chat_completion.choices[0].message.content

    except Exception as e:
        print(f"AI ERROR: {e}")
        ai_reply = "Oops! Something went wrong on my end 😵 Try again!"

    # Save AI reply
    data["history"].append({"role": "assistant", "content": ai_reply})
    save_memory()

    try:
    
        sent_msg = await update.message.reply_text(str(ai_reply)[:3000])
        if user_id not in bot_messages:
            bot_messages[user_id] = []
        bot_messages[user_id].append(sent_msg.message_id)

        # Voice reply if voice_mode is ON
        if data.get("voice_mode", False):
            await context.bot.send_chat_action(chat_id=user_id, action=ChatAction.RECORD_VOICE)
            await send_voice_reply(update, ai_reply, lang="en")

    except Exception as e:
        print(f"SEND ERROR: {e}")


# ─── APP SETUP ────────────────────────────────────────────────────────────────

app = (
    ApplicationBuilder()
    .token(BOT_TOKEN)
    .connect_timeout(60)
    .read_timeout(60)
    .write_timeout(60)
    .pool_timeout(60)
    .build()
)

# Commands
app.add_handler(CommandHandler("start", start))
app.add_handler(CommandHandler("clear", clear))
app.add_handler(CommandHandler("help", help_command))
app.add_handler(CommandHandler("mode", mode_command))
app.add_handler(CommandHandler("voicemode", voicemode_command))
app.add_handler(CommandHandler("joke", joke))
app.add_handler(CommandHandler("motivate", motivate))
app.add_handler(CommandHandler("search", search))

# Media handlers
app.add_handler(
    MessageHandler(
        filters.PHOTO & ~filters.COMMAND,
        handle_photo
    )
)
app.add_handler(MessageHandler(filters.VOICE, handle_voice))
app.add_handler(MessageHandler(filters.Document.ALL, handle_document))

# Text messages
app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, reply))

print("🤖 AI Bot is running...")
app.run_polling(
    drop_pending_updates=True,
    allowed_updates=Update.ALL_TYPES,
    timeout=30,
    bootstrap_retries=10
)