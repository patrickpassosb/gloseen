# -*- coding: utf-8 -*-
# Main application for Vision AI Pro
from flask import Flask, render_template, jsonify, request
import cv2
import numpy as np
import requests
import time
import os
import glob
import sqlite3
import json
import base64
from datetime import datetime
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# --- IA IMPORTS ---
from ultralytics import YOLO
from azure.cognitiveservices.vision.computervision import ComputerVisionClient
from msrest.authentication import CognitiveServicesCredentials
import azure.cognitiveservices.speech as speechsdk
from azure.cognitiveservices.vision.face import FaceClient
from openai import AzureOpenAI

app = Flask(__name__, static_folder='static')

@app.template_filter('from_json')
def from_json_filter(s):
    try: return json.loads(s)
    except: return []

# ==========================================
# SETTINGS AND CONSTANTS
# ==========================================
# Azure Credentials from environment
AZURE_VISION_ENDPOINT = os.getenv("AZURE_VISION_ENDPOINT")
AZURE_VISION_KEY = os.getenv("AZURE_VISION_KEY")
AZURE_FACE_ENDPOINT = os.getenv("AZURE_FACE_ENDPOINT")
AZURE_FACE_KEY = os.getenv("AZURE_FACE_KEY")
AZURE_OPENAI_KEY = os.getenv("AZURE_OPENAI_KEY")
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_DEPLOYMENT_NAME = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME")
AZURE_SPEECH_KEY = os.getenv("AZURE_SPEECH_KEY")
AZURE_SPEECH_REGION = os.getenv("AZURE_SPEECH_REGION")

URL_ESP32 = os.getenv("URL_ESP32", "http://192.168.15.9/capture")
TEMP_FOLDER = "static"
if not os.path.exists(TEMP_FOLDER): os.makedirs(TEMP_FOLDER)

DB_NAME = "vision_memory.db"
# File names for audio
EN_AUDIO_FILENAME = "audio_en.wav"
RESPONSE_AUDIO_FILENAME = "audio_response.wav"
CUE_AUDIO_FILENAME = "can_speak.wav"
ALARM_AUDIO_FILENAME = "alarm.wav"

EN_AUDIO_PATH = os.path.join(TEMP_FOLDER, EN_AUDIO_FILENAME)
RESPONSE_AUDIO_PATH = os.path.join(TEMP_FOLDER, RESPONSE_AUDIO_FILENAME)
CUE_AUDIO_PATH = os.path.join(TEMP_FOLDER, CUE_AUDIO_FILENAME)
ALARM_AUDIO_PATH = os.path.join(TEMP_FOLDER, ALARM_AUDIO_FILENAME)

# --- NEURAL VOICES ---
EN_VOICE = 'en-US-JennyNeural'

# --- DANGER LIST (English) ---
IMMEDIATE_DANGER_WORDS = [
    'knife', 'scissors', 'gun', 'fire', 'flame', 'explosive',
    'smoke', 'vapor', 'stairs', 'step', 'sharp edge', 'hole', 'obstacle', 'gap', 'uneven surface'
]

LAST_ANALYSIS_CONTEXT = None

print("--- INITIALIZING VISION AI PRO (EN) ---")
try:
    computervision_client = ComputerVisionClient(AZURE_VISION_ENDPOINT, CognitiveServicesCredentials(AZURE_VISION_KEY))
    face_client = FaceClient(AZURE_FACE_ENDPOINT, CognitiveServicesCredentials(AZURE_FACE_KEY))
    openai_client = AzureOpenAI(api_key=AZURE_OPENAI_KEY, api_version="2024-02-01", azure_endpoint=AZURE_OPENAI_ENDPOINT)
    model = YOLO('yolov8x.pt')
    print("All Azure IAs Loaded.")
except Exception as e: print(f"Azure IA Error: {e}")

# --- DATABASE FUNCTIONS ---
def get_db_connection(): conn = sqlite3.connect(DB_NAME); conn.row_factory = sqlite3.Row; return conn
def init_db():
    with get_db_connection() as conn: conn.execute('''CREATE TABLE IF NOT EXISTS history (id INTEGER PRIMARY KEY AUTOINCREMENT, timestamp TEXT, datetime_formatted TEXT, image_path TEXT, gpt_summary TEXT, objects_json TEXT, has_danger INTEGER)'''); conn.commit()
init_db()

# --- STATIC AUDIO GENERATORS ---
def generate_static_audios():
    speech_config = speechsdk.SpeechConfig(subscription=AZURE_SPEECH_KEY, region=AZURE_SPEECH_REGION)
    speech_config.speech_synthesis_voice_name='en-US-GuyNeural' # Changed default voice to EN
    if not os.path.exists(CUE_AUDIO_PATH):
        try:
            audio_config = speechsdk.audio.AudioOutputConfig(filename=CUE_AUDIO_PATH)
            synthesizer = speechsdk.SpeechSynthesizer(speech_config=speech_config, audio_config=audio_config)
            synthesizer.speak_text_async("You can speak now.").get()
        except: pass
    if not os.path.exists(ALARM_AUDIO_PATH):
        try:
            audio_config = speechsdk.audio.AudioOutputConfig(filename=ALARM_AUDIO_PATH)
            synthesizer = speechsdk.SpeechSynthesizer(speech_config=speech_config, audio_config=audio_config)
            synthesizer.speak_text_async("Critical danger! Attention!").get() 
        except: pass
generate_static_audios()

# --- HELPER FUNCTIONS ---
def cleanup_old_images():
    try:
        for arq in glob.glob(os.path.join(TEMP_FOLDER, "capture_*.jpg")): os.remove(arq)
        for arq in glob.glob(os.path.join(TEMP_FOLDER, "processed_*.jpg")): os.remove(arq)
    except: pass

# --- NEW SIMPLE AND SECURE AUDIO GENERATION FUNCTION ---
def generate_simple_neural_audio(text, voice, output_file, web_name):
    """Generates a simple audio file with a single voice. Much more robust."""
    if not text: return None
    try:
        # Delete previous file if it exists to ensure no old audio is played
        if os.path.exists(output_file):
            try: os.remove(output_file)
            except: pass # Ignore error if file is in use, Azure will overwrite

        speech_config = speechsdk.SpeechConfig(subscription=AZURE_SPEECH_KEY, region=AZURE_SPEECH_REGION)
        speech_config.speech_synthesis_voice_name = voice
        audio_config = speechsdk.audio.AudioOutputConfig(filename=output_file)
        synthesizer = speechsdk.SpeechSynthesizer(speech_config=speech_config, audio_config=audio_config)
        
        result = synthesizer.speak_text_async(text).get()
        
        if result.reason == speechsdk.ResultReason.SynthesizingAudioCompleted:
            # Add timestamp to avoid browser cache
            return f"/static/{web_name}?t={int(time.time())}_{np.random.randint(0,1000)}"
        else:
            print(f"Azure TTS Error ({voice}): {result.cancellation_details.reason}")
            return None
    except Exception as e:
        print(f"TTS Exception: {e}")
        return None

# --- GPT-4o VISION (ENVIRONMENTAL HAZARDS) ---
def analyze_image_with_gpt4o_vision(image_path):
    try:
        with open(image_path, "rb") as image_file: encoded_image = base64.b64encode(image_file.read()).decode('utf-8')
        system_prompt = """You are a CRITICAL safety vision system.
        YOUR PRIORITY MISSION: Identify physical hazards in the environment and dangerous objects.
        DETECTION INSTRUCTIONS:
        1. ENVIRONMENTAL HAZARDS (Highest Priority): Obsessively look for: 'stairs', 'steps', 'sharp edges', 'holes', 'uneven surfaces', 'smoke', 'steam', 'obstacles'.
        2. DANGEROUS OBJECTS: Identify 'knife', 'gun', 'fire'.
        3. PERSONAL OBJECTS: Identify cell phones, keys, glasses.
        Return ONLY a list of identified objects/hazards, separated by commas, in English."""
        response = openai_client.chat.completions.create(
            model=AZURE_OPENAI_DEPLOYMENT_NAME,
            messages=[{"role": "system", "content": system_prompt},
                      {"role": "user", "content": [{"type": "text", "text": "Analyze the image for environmental hazards and objects."},
                                                   {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{encoded_image}"}}]}],
            temperature=0.0, max_tokens=300
        )
        return response.choices[0].message.content
    except Exception as e: print(f"❌ GPT Vision Error: {e}"); return "Error in visual analysis."

# --- INTELLIGENT SUMMARY ---
def generate_intelligent_summary(scene_data, alert_status):
    en_instruction = ""
    if alert_status == "critico":
        en_instruction = "CRITICAL ALERT: An alarm has sounded. Start with a direct warning about the danger detected."

    system_prompt = f"""You are a visual assistant (English).
    Use 'DETECCAO_VISUAL_GPT_VISION' as base.
    TASK: Generate a concise and helpful summary in English.
    EN Guidelines: {en_instruction} Be direct.
    """
    user_prompt = f"""Scene Data: {scene_data}. Generate the summary."""
    try:
        response = openai_client.chat.completions.create(
            model=AZURE_OPENAI_DEPLOYMENT_NAME, messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}],
            temperature=0.3, max_tokens=600
        )
        return response.choices[0].message.content.strip()
    except: return "Error in summary."

# --- WEB ROUTES ---
@app.route('/')
def index(): return render_template('index.html')

@app.route('/history')
def history():
    conn = get_db_connection(); captures = conn.execute('SELECT * FROM history ORDER BY id DESC LIMIT 20').fetchall(); conn.close()
    return render_template('memory.html', capturas=captures)

@app.route('/analyze', methods=['POST'])
def analyze():
    global LAST_ANALYSIS_CONTEXT
    print(">>> STARTING ANALYSIS (HAZARDS + SEPARATE AUDIOS) <<<")
    cleanup_old_images(); current_timestamp = int(time.time()); formatted_datetime = datetime.now().strftime("%d/%m/%Y %H:%M")
    filename = f"capture_{current_timestamp}.jpg"; image_path = os.path.join(TEMP_FOLDER, filename)

    # ESP32 Capture
    try:
        response = requests.get(URL_ESP32, timeout=10)
        if response.status_code == 200 and len(response.content) > 0:
            with open(image_path, "wb") as f: f.write(response.content); f.flush(); os.fsync(f.fileno()); time.sleep(0.2)
        else: return jsonify({'status': 'error', 'message': 'ESP32 capture error.'})
    except Exception as e: return jsonify({'status': 'error', 'message': f'ESP32 failure: {e}'})

    # AI Execution
    print("SENDING TO GPT-4o VISION...")
    gpt_visual_detection = analyze_image_with_gpt4o_vision(image_path)

    # Critical Alert
    alert_status = "none"
    if any(hazard in gpt_visual_detection.lower() for hazard in IMMEDIATE_DANGER_WORDS): alert_status = "critico"

    brain_data = {'DETECCAO_VISUAL_GPT_VISION': gpt_visual_detection, 'STATUS_ALERTA_ATUAL': alert_status}
    LAST_ANALYSIS_CONTEXT = brain_data

    # --- AUDIO GENERATION ---
    text_en = generate_intelligent_summary(brain_data, alert_status)
    
    # Generate EN audio
    en_audio_url = generate_simple_neural_audio(text_en, EN_VOICE, EN_AUDIO_PATH, EN_AUDIO_FILENAME)
    
    # Save to history
    try:
        object_list = [item.strip() for item in gpt_visual_detection.split(',') if item.strip()]
        with get_db_connection() as conn: conn.execute('''INSERT INTO history VALUES (NULL, ?, ?, ?, ?, ?, ?)''', (str(current_timestamp), formatted_datetime, f"/static/{filename}", text_en, json.dumps(object_list, ensure_ascii=False), 1 if alert_status != "none" else 0)); conn.commit()
    except: pass

    # RETURNS DATA
    return jsonify({
        'status': 'success',
        'yolo': gpt_visual_detection, 
        'azure_desc': "GPT-4o Vision Analysis (Environment+Objects)", 
        'status_alerta': alert_status,
        'image_url': f"/static/{filename}",
        'audio_url_en': en_audio_url
    })

@app.route('/ask', methods=['POST'])
def ask():
    global LAST_ANALYSIS_CONTEXT
    if LAST_ANALYSIS_CONTEXT is None: return jsonify({'status': 'error', 'response': 'Analyze first.'})
    data = request.get_json(); question = data.get('question')
    if not question: return jsonify({'status': 'error', 'response': 'Empty.'})
    
    # Always respond in English for the final version
    lang_instruction = "Respond in English."
    
    # Generate response
    try:
        response = openai_client.chat.completions.create(
            model=AZURE_OPENAI_DEPLOYMENT_NAME, messages=[{"role": "system", "content": f"Use 'DETECCAO_VISUAL_GPT_VISION'. {lang_instruction} Be brief."}, {"role": "user", "content": f"Context: {LAST_ANALYSIS_CONTEXT}. Question: {question}"}],
            temperature=0.5, max_tokens=200
        )
        response_text = response.choices[0].message.content
    except: response_text = "Error.";

    # Generate audio response with English voice
    response_audio_url = generate_simple_neural_audio(response_text, EN_VOICE, RESPONSE_AUDIO_PATH, RESPONSE_AUDIO_FILENAME)

    return jsonify({'status': 'success', 'response_text': response_text, 'audio_url': response_audio_url})

@app.route('/read_text', methods=['POST'])
def read_text(): return jsonify({'status':'error'})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)