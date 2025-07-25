import grpc
from concurrent import futures
import ai_service_pb2
import ai_service_pb2_grpc
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from mistralai import Mistral
import os
import torch
import logging
import socket
import re
from accelerate import infer_auto_device_map
import pyttsx3
import win32com.client
import threading
import queue
from datetime import datetime
import time
import requests
import geocoder
import sqlite3
from gtts import gTTS
from pydub import AudioSegment
from pydub.playback import play
import tempfile
from langdetect import detect
from pydub.utils import which
AudioSegment.converter = which("C:/ffmpeg/bin/ffmpeg.exe")# Add to existing imports
import speech_recognition as sr
import io

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AIService(ai_service_pb2_grpc.AIServiceServicer):
    def __init__(self):
        # Configuration
        self.model_name = "WizardLMTeam/WizardMath-7B-V1.1"
        self.mistral_api_key = "ZIHMju2yT1xWtZ8MUexSMjQcmtEgQRqI"  # Replace with your key
        self.mistral_model = "mistral-large-latest"  # Or "mistral-medium", "mistral-small"
        self.weather_api_key = "10385f3a2ad33ef0b91d9672d48d7755" # openweather_key
        
        # Initialize with GPU optimization
        self.wizardmath_model = None
        self.wizardmath_tokenizer = None
        self._initialize_models()
        
        # Initialize Mistral client
        self.mistral_client = Mistral(api_key=self.mistral_api_key)

        # Add this voice engine initialization
        self.voice_engine = pyttsx3.init()

        # Initialize voice engine
        # self.voice_engine = None
        self._initialize_voice_engine()

        # Speak startup message
        #self._announce_system_status()

        # Initialize database for long-term memory
        self.db_conn = sqlite3.connect('ai_memory.db', check_same_thread=False)
        self._init_db()

        # Short-term memory (in-memory)
        self.current_session = {
            "mistral": [],
            "wizardmath": []
        }
        
        # Memory configuration
        self.max_tokens = 3000
        self.summary_frequency = 3  # Summarize every 3 exchanges
        self.max_summaries_to_keep = 100  # Limit database size

    def _initialize_models(self):
        """Initialize models with maximum GPU optimization"""
        try:
            # Verify CUDA
            if not torch.cuda.is_available():
                raise RuntimeError("CUDA not available - check your PyTorch installation")
            
            logger.info(f"Using GPU: {torch.cuda.get_device_name(0)}")
            logger.info(f"Available GPU Memory: {torch.cuda.get_device_properties(0).total_memory/1024**3:.2f}GB")

            # Configure 4-bit quantization
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True
            )

            # Calculate optimal device map
            max_memory = {0: "7GB", "cpu": "10GB"}  # Leave 1GB GPU buffer
            
            # Initialize tokenizer
            self.wizardmath_tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                trust_remote_code=True
            )
            self.wizardmath_tokenizer.pad_token = self.wizardmath_tokenizer.eos_token

            # Load model with memory optimization
            self.wizardmath_model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                trust_remote_code=True,
                quantization_config=quantization_config,
                device_map="auto",
                max_memory=max_memory,
                torch_dtype=torch.float16,
                offload_folder="./offload",
                low_cpu_mem_usage=True
            ).eval()

            logger.info(f"Model loaded on devices: {self.wizardmath_model.hf_device_map}")
            logger.info(f"GPU memory allocated: {torch.cuda.memory_allocated()/1024**3:.2f}GB")

        except Exception as e:
            logger.error(f"Model initialization failed: {e}")
            raise RuntimeError("Failed to initialize model with GPU optimization")

    def _initialize_voice_engine(self):
        """Initialize voice settings with proper speed control"""
        try:
            self.voice_settings = {
                'voice_id': None,
                'rate': 150,  # Much slower speech rate (default is 200)
                'volume': 0.95,
                'engine_type': 'pyttsx3',  # default
                'pauses': {
                    'sentence': 0.05,  # Longer pause between sentences
                    'comma': 0.05,    # Pause after commas
                    'colon': 0.1      # Pause after colons/semicolons
                },
                'use_gtts': True  # Enable Google TTS for Punjabi
            }

            # Try Windows SAPI first (more reliable)
            try:
                temp_engine = win32com.client.Dispatch("SAPI.SpVoice")
                voices = temp_engine.GetVoices()
            
                preferred_voices = [
                    "Microsoft David", "Microsoft Hazel", 
                    "Microsoft George", "IVONA 2 Brian"
                ]
            
                for voice_name in preferred_voices:
                    for voice in voices:
                        if voice_name in voice.GetDescription():
                            self.voice_settings['voice'] = voice
                            self.voice_settings['engine_type'] = 'sapi'
                            self.voice_settings['rate'] = 0.5  # Slower than normal (-10 to 10)
                            break
            
                logger.info("Initialized Windows SAPI voice settings")
                return
            except Exception as e:
                logger.info(f"Couldn't initialize SAPI: {e}, falling back to pyttsx3")

            # Fall back to pyttsx3
            try:
                temp_engine = pyttsx3.init()
                voices = temp_engine.getProperty('voices')
            
                for voice in voices:
                    if 'english' in voice.id.lower() and 'male' in voice.id.lower():
                        self.voice_settings['voice_id'] = voice.id
                        break
            
                logger.info("Initialized pyttsx3 voice settings")
            except Exception as e:
                logger.error(f"Failed to initialize any voice engine: {e}")
                self.voice_settings = None
        except Exception as e:
            logger.error(f"Voice initialization error: {e}")
            self.voice_settings = None

    def RecognizeSpeech(self, request, context):
        """New gRPC method for speech recognition"""
        try:
            # Convert gRPC audio bytes to WAV
            audio = AudioSegment.from_raw(
                io.BytesIO(request.audio_data),
                sample_width=2,
                frame_rate=request.sample_rate,
                channels=1
            )
        
            # Export to WAV format
            wav_io = io.BytesIO()
            audio.export(wav_io, format="wav")
        
            # Recognize using Google Web API (most accurate free option)
            r = sr.Recognizer()
            with sr.AudioFile(wav_io) as source:
                audio_data = r.record(source)
                text = r.recognize_google(
                    audio_data,
                    language=request.language_code
                )
        
            return ai_service_pb2.QueryResponse(
                response_text=text,
                ai_source="SpeechRecognition"
            )
        
        except Exception as e:
            logger.error(f"Speech recognition failed: {e}")
            return ai_service_pb2.QueryResponse(
                response_text="Could not understand audio",
                ai_source="System"
            )

    def _speak_punjabi(self, text):
        """Robust Punjabi TTS with proper cleanup"""
        temp_path = None
        try:
            # Create temp file
            with tempfile.NamedTemporaryFile(suffix='.mp3', delete=False) as tmp:
                temp_path = tmp.name
        
            # Generate speech
            tts = gTTS(text=text, lang='pa', slow=False)
            tts.save(temp_path)
        
            # Verify file
            if not os.path.exists(temp_path) or os.path.getsize(temp_path) == 0:
                raise RuntimeError("Empty audio file generated")

            # Play audio synchronously
            sound = AudioSegment.from_mp3(temp_path)
            play(sound)  # This will block until playback completes
        
        except Exception as e:
            logger.error(f"Punjabi TTS failed: {e}")
            # Fallback to English
            self._speak_response(f"[Punjabi unavailable] {text}")
        finally:
            if temp_path and os.path.exists(temp_path):
                try:
                    os.unlink(temp_path)
                except:
                    pass
            

    def GetSystemStatus(self, request, context):
        """gRPC endpoint for system status and weather"""
        greeting = self._get_time_based_greeting()
        location, coords = self._get_location()
        weather = self._get_weather(*coords)
    
        # Cross-platform time formatting
        now = datetime.now()
        hour = now.hour % 12
        if hour == 0:
            hour = 12
        ampm = "a.m." if now.hour < 12 else "p.m."
        current_time = f"{hour}:{now.minute:02d} {ampm}"

        if not weather:
            return ai_service_pb2.SystemStatus(
                greeting=greeting,
                weather_report="Weather systems offline",
                location=location,
                local_time=datetime.now().strftime("%H:%M")
            )

        # Convert wind speed to more natural phrasing / Not using currently
        wind_speed = weather['wind']
        if wind_speed < 0.3:
            wind_desc = "calm conditions"
        elif wind_speed < 1.5:
            wind_desc = f"a light breeze at {wind_speed} meters per second"
        else:
            wind_desc = f"winds of {wind_speed} meters per second"

        # Build the detailed weather report
        detailed_report = (
            f"{greeting}, it's {current_time}. The weather in {location} is {weather['temp']}°C "
            f"with {weather['condition']}. Humidity is at {weather['humidity']}% "
            f"and wind speed is {wind_speed} meters per second."
        )

        return ai_service_pb2.SystemStatus(
            greeting=greeting,
            weather_report=detailed_report,
            location=location,
            temperature=weather['temp'],
            conditions=weather['condition'],
            wind_speed=weather['wind'],
            humidity=weather['humidity'],
            local_time=datetime.now().strftime("%H:%M")
        )

    def _announce_system_status(self):
        """J.A.R.V.I.S-style startup announcement"""
        status = self.GetSystemStatus(ai_service_pb2.Empty(), None)
        message = status.weather_report
        print(f"J.A.R.V.I.S: {message}")
        self._speak_response(message)

    def _get_time_based_greeting(self):
        hour = datetime.now().hour
        if 5 <= hour < 12:
            return "Good morning"
        elif 12 <= hour < 17:
            return "Good afternoon"
        elif 17 <= hour < 21:
            return "Good evening"
        else:
            return "Good night"

    def _get_location(self):
        try:
            g = geocoder.ip('me')
            return g.city, g.latlng
        except Exception:
            return "Malibu", (34.0259, -118.7798)  # Default to Tony Stark's home

    def _get_weather(self, lat, lng):
            try:
                url = f"https://api.openweathermap.org/data/2.5/weather?lat={lat}&lon={lng}&appid={self.weather_api_key}&units=metric"
                response = requests.get(url, timeout=3)
                response.raise_for_status()
                data = response.json()
                return {
                    'temp': data['main']['temp'],
                    'condition': data['weather'][0]['description'],
                    'humidity': data['main']['humidity'],
                    'wind': data['wind']['speed'],
                    'location': data.get('name', 'Current location')
                }
            except Exception as e:
                logger.error(f"Weather API error: {e}")
                return None

    def _jarvis_style_response(self, text, ai_source):
        """Format responses with J.A.R.V.I.S mannerisms"""
        # Add appropriate prefixes based on query type
        if ai_source == "WizardMath": #"DeepSeek-Math":
            prefixes = ["The solution appears to be", 
                       "Calculations complete:",
                       "Mathematically speaking,"]
        else:
            prefixes = ["According to my analysis,",
                       "My research indicates,",
                       "I would suggest,"]
    
        # Randomly select a prefix
        import random
        prefix = random.choice(prefixes)
    
        # Clean up the response text
        text = text.strip()
        if not text.endswith(('.','!','?')):
            text += '.'
    
        return f"{prefix} {text[0].lower() + text[1:]}"  # Lowercase first letter after prefix

    def _speak_response(self, text):
        """Speak the response with proper pacing and pauses"""
        if not self.voice_settings:
            print(f"\nJ.A.R.V.I.S: {text}\n")
            return

        # Print immediately
        print(f"\nJ.A.R.V.I.S: {text}\n")


        # Split text into segments by language
        segments = self._split_text_by_language(text)

        # Start speech in a thread
        def speak_job():
            try:
                    # First stop any ongoing speech
                if self.voice_settings['engine_type'] == 'pyttsx3':
                    self.voice_engine.stop()
                elif hasattr(self, '_last_tts_process'):
                    self._last_tts_process.terminate()
            
                time.sleep(1.0)  # Ensure complete stop

                for i, (text_part, lang) in enumerate(segments):
                    if not text_part.strip():
                        continue

                    if lang == 'pa':
                        # For Punjabi, use gTTS
                        self._speak_punjabi(text_part)
                    else:
                        # For English, use the configured engine
                        if self.voice_settings['engine_type'] == 'sapi':
                            engine = win32com.client.Dispatch("SAPI.SpVoice")
                            engine.Rate = self.voice_settings['rate']
                            engine.Speak(text_part)
                        else:
                            engine = pyttsx3.init()
                            engine.setProperty('rate', self.voice_settings['rate'])
                            engine.say(text_part)
                            engine.runAndWait()
                
                    # Add pause between segments but not after last one
                    if i < len(segments) - 1:
                        time.sleep(1.35)  # 1 second pause between language changes

            except Exception as e:
                logger.error(f"Speech failed: {e}")
                    
                

        # Start the speech thread
        threading.Thread(target=speak_job, daemon=True).start()

    def _split_text_by_language(self, text):
        """Improved language segmentation that keeps punctuation with words"""
        segments = []
        current_lang = 'en'  # Default to English
        current_segment = []
        punjabi_range = range(0x0A00, 0x0A7F)  # Gurmukhi Unicode range

        i = 0
        while i < len(text):
            char = text[i]
        
            # Check if character is Punjabi
            is_punjabi = ord(char) in punjabi_range if char.strip() else False
        
            # Handle language transitions
            if is_punjabi and current_lang != 'pa':
                if current_segment:
                    segments.append((''.join(current_segment), current_lang))
                current_segment = []
                current_lang = 'pa'
            elif not is_punjabi and current_lang == 'pa' and char.strip():
                # Look ahead to see if this is a punctuation after Punjabi
                if i+1 < len(text) and not text[i+1].strip():
                    # If next character is whitespace/punctuation, keep in Punjabi segment
                    pass
                else:
                    if current_segment:
                        segments.append((''.join(current_segment), current_lang))
                    current_segment = []
                    current_lang = 'en'
        
            current_segment.append(char)
            i += 1

        if current_segment:
            segments.append((''.join(current_segment), current_lang))
    
        return segments

    def _init_db(self):
            """Initialize the database tables"""
            cursor = self.db_conn.cursor()
        
            # Create tables if they don't exist
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS conversation_memory (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    model_type TEXT NOT NULL,
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            ''')
        
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS memory_summaries (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    model_type TEXT NOT NULL,
                    summary TEXT NOT NULL,
                    original_count INTEGER NOT NULL,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            ''')
        
            # Create index for faster queries
            cursor.execute('''
                CREATE INDEX IF NOT EXISTS idx_model_type ON memory_summaries (model_type)
            ''')
        
            self.db_conn.commit()

    def _add_to_memory(self, model_type, role, content):
            """Add conversation to memory with automatic summarization"""
            try:
                # Add to current session (short-term memory)
                self.current_session[model_type].append({
                    "role": role,
                    "content": content,
                    "timestamp": datetime.now().isoformat()
                })
            
                # Store in database (long-term raw memory)
                cursor = self.db_conn.cursor()
                cursor.execute('''
                    INSERT INTO conversation_memory (model_type, role, content)
                    VALUES (?, ?, ?)
                ''', (model_type, role, content))
                self.db_conn.commit()
            
                # Check if we need to summarize
                if len(self.current_session[model_type]) >= self.summary_frequency:
                    self._summarize_conversation(model_type)
                
            except Exception as e:
                logger.error(f"Failed to add to memory: {e}")

    def _summarize_conversation(self, model_type):
            """Create and store a summary of recent conversation"""
            try:
                recent_chat = self.current_session[model_type]
                if not recent_chat:
                    return
                
                # Prepare conversation text for summarization
                conversation_text = "\n".join(
                    f"{msg['role']}: {msg['content']}" 
                    for msg in recent_chat
                )
            
                # Get summary from Mistral
                summary_prompt = f"""Please summarize the key points from this conversation in 3-4 bullet points:
            
                {conversation_text}
            
                Summary:
                - """
            
                summary_response = self.mistral_client.chat.complete(
                    model="mistral-small",
                    messages=[{"role": "user", "content": summary_prompt}],
                    max_tokens=150
                )
            
                summary = summary_response.choices[0].message.content
            
                # Store summary in database
                cursor = self.db_conn.cursor()
                cursor.execute('''
                    INSERT INTO memory_summaries (model_type, summary, original_count)
                    VALUES (?, ?, ?)
                ''', (model_type, summary, len(recent_chat)))
                self.db_conn.commit()
            
                # Clear current session after summarization
                self.current_session[model_type] = []
            
                # Enforce max summaries limit
                self._cleanup_old_summaries(model_type)
            
            except Exception as e:
                logger.error(f"Summarization failed: {e}")

    def _cleanup_old_summaries(self, model_type):
            """Remove oldest summaries to maintain database size limit"""
            try:
                cursor = self.db_conn.cursor()
            
                # Get count of summaries for this model
                cursor.execute('''
                    SELECT COUNT(*) FROM memory_summaries 
                    WHERE model_type = ?
                ''', (model_type,))
                count = cursor.fetchone()[0]
            
                if count > self.max_summaries_to_keep:
                    # Delete oldest entries exceeding the limit
                    cursor.execute('''
                        DELETE FROM memory_summaries
                        WHERE id IN (
                            SELECT id FROM memory_summaries
                            WHERE model_type = ?
                            ORDER BY timestamp ASC
                            LIMIT ?
                        )
                    ''', (model_type, count - self.max_summaries_to_keep))
                    self.db_conn.commit()
                
            except Exception as e:
                logger.error(f"Failed to clean up old summaries: {e}")

    def _get_recent_summaries(self, model_type, limit=2):
            """Retrieve most recent summaries from database"""
            try:
                cursor = self.db_conn.cursor()
                cursor.execute('''
                    SELECT summary FROM memory_summaries
                    WHERE model_type = ?
                    ORDER BY timestamp DESC
                    LIMIT ?
                ''', (model_type, limit))
            
                return [row[0] for row in cursor.fetchall()]
            except Exception as e:
                logger.error(f"Failed to fetch summaries: {e}")
                return []

    def _get_context(self, model_type, new_query):
            """Build context from both short-term and long-term memory"""
            # Get recent messages from current session
            recent_messages = self.current_session[model_type].copy()
        
            # Get relevant summaries from database
            summaries = self._get_recent_summaries(model_type)
        
            # Prepare context messages
            context_messages = []
        
            # Add summaries as system messages
            for summary in summaries:
                context_messages.append({
                    "role": "system",
                    "content": f"Previous conversation summary: {summary}"
                })
        
            # Add recent messages
            for msg in recent_messages:
                context_messages.append({
                    "role": msg["role"],
                    "content": msg["content"]
                })
        
            # Add new query
            context_messages.append({
                "role": "user",
                "content": new_query
            })
        
            return context_messages

    def ProcessQuery(self, request, context):
        """Process queries with full J.A.R.V.I.S personality"""
        try:
            # Display user input
            print(f"\nUSER: {request.input_text}")

            # Detect if input is Punjabi
            is_punjabi = False
            try:
                is_punjabi = detect(request.input_text) == 'pa'
            except:
                pass
        
            # Get response
            if self._is_math_or_code_query(request.input_text):
                raw_response = self._process_with_wizardmath(request.input_text)
                ai_source = "WizardMath"
            else:
                raw_response = self._process_with_mistral(request.input_text)
                ai_source = "Mistral"
        
            # Apply J.A.R.V.I.S formatting
            response = self._jarvis_style_response(raw_response, ai_source)

            # Special handling for Punjabi responses
            if is_punjabi:
                response = response.replace("sir", "ਜੀ")  # Replace "sir" with Punjabi equivalent
        
            # Display and speak simultaneously
            print(f"J.A.R.V.I.S: {response}")
            self._speak_response(response)
        
            return ai_service_pb2.QueryResponse(
                response_text=response,
                ai_source=ai_source
                #is_punjabi=is_punjabi  # Optional: Add field to proto if needed
            )
        
        except Exception as e:
            error_msg = "My apologies, sir. I seem to be experiencing a system irregularity."
            print(f"J.A.R.V.I.S: {error_msg}")
            self._speak_response(error_msg)
            return ai_service_pb2.QueryResponse(
                response_text=error_msg,
                ai_source="System"
            )

    def _is_math_or_code_query(self, text):
        """More precise routing between DeepSeek and Mistral"""
        text_lower = text.lower()
    
        # Keywords that DEFINITELY require DeepSeek
        math_keywords = ["solve", "calculate", "equation", "integral", "derivative", "x=", "wizardmath"]
        # code_keywords = ["code", "python", "function", "algorithm", "programming", "loop"]
    
        # Questions that should go to Mistral despite keywords
        mistral_exceptions = [
            "what is", "explain", "tell me about", "history of", 
            "who invented", "why is", "how does"
        ]
    
        # Check for exceptions first
        if any(phrase in text_lower for phrase in mistral_exceptions):
            return False
    
        # Default to DeepSeek for math/code
        return any(keyword in text_lower for keyword in math_keywords) # + code_keywords)

    def _process_with_wizardmath(self, prompt):
        """Process math queries with WizardMath with reliable answer extraction"""
        # Use a more structured prompt template
        template = """Please solve the following math problem carefully:
    
        Problem: {prompt}
    
        Requirements:
        1. Show each calculation step clearly
        2. Box the final answer like this: \boxed{{answer}}
        3. Verify your calculations
    
        Let's begin:
        """
    
        inputs = self.wizardmath_tokenizer(template.format(prompt=prompt), 
                    return_tensors="pt").to(self.wizardmath_model.device)
        outputs = self.wizardmath_model.generate(
            **inputs,
            max_new_tokens=200,
            do_sample=False,
            temperature=0.1,  # Completely deterministic
            pad_token_id=self.wizardmath_tokenizer.eos_token_id
        )
    
        full_response = self.wizardmath_tokenizer.decode(outputs[0], skip_special_tokens=True)
    
        # Robust answer extraction methods
        answer = None
    
        # Method 1: Check for boxed answer \boxed{...}
        boxed_pattern = r'\\boxed\{([^}]*)\}'
        boxed_match = re.search(boxed_pattern, full_response)
        if boxed_match:
            answer = boxed_match.group(1).strip()
    
        # Method 2: Check for "Final Answer:" pattern
        if not answer and "Final Answer:" in full_response:
            answer_part = full_response.split("Final Answer:")[1].strip()
            answer = answer_part.split('\n')[0].split('.')[0].strip()
    
        # Method 3: Extract last number in response
        if not answer:
            numbers = re.findall(r'-?\d+\.?\d*', full_response)
            if numbers:
                answer = numbers[-1]
    
        # Final fallback
        if not answer:
            answer = "Could not determine the answer"
    
        return answer

    def _process_with_mistral(self, prompt):
        """Enhanced Mistral processing with memory integration"""
        try:
            # Step 1: Get context from memory system
            messages = self._get_context("mistral", prompt)
        
            # Debug logging (optional)
            logger.debug(f"Mistral API messages context:\n{messages}")
        
            # Step 2: Call Mistral API with full context
            chat_response = self.mistral_client.chat.complete(
                model=self.mistral_model,
                messages=messages,
                temperature=0.7,  # Added for more controlled responses
                max_tokens=500    # Added to prevent cutoff responses
            )
        
            response_content = chat_response.choices[0].message.content
        
            # Step 3: Store exchange in memory
            self._add_to_memory("mistral", "user", prompt)
            self._add_to_memory("mistral", "assistant", response_content)
        
            # Step 4: Return the generated response
            return response_content
        
        except Exception as e:
            logger.error(f"Mistral API error: {e}")
        
            # Store the failed query for debugging (without response)
            self._add_to_memory("mistral", "user", f"[FAILED QUERY] {prompt}")
        
            return "I couldn't process your request. Please try again later."

def find_available_port(start_port=50051, max_tries=100):
    """Find an available port starting from start_port"""
    for port in range(start_port, start_port + max_tries):
        try:
            with socket.socket(socket.AF_INET6, socket.SOCK_STREAM) as s:
                s.bind(('::', port))
                return port
        except OSError:
            continue
    raise RuntimeError(f"No available ports found in range {start_port}-{start_port + max_tries - 1}")

def serve():
    """Start the gRPC server with robust port handling"""
    # Configure server with port reuse
    server = grpc.server(
        futures.ThreadPoolExecutor(max_workers=10),
        options=(('grpc.so_reuseport', 1),)  # Allow port reuse
    )
    ai_service_pb2_grpc.add_AIServiceServicer_to_server(AIService(), server)
    
    port = find_available_port()
    server.add_insecure_port(f'[::]:{port}')
    
    try:
        server.start()
        logger.info(f"Server started successfully on port {port} (PID: {os.getpid()})")
        logger.info("Python gRPC server is running...")
        server.wait_for_termination()
    except KeyboardInterrupt:
        logger.info("Shutting down server gracefully...")
        server.stop(0)
    except Exception as e:
        logger.error(f"Server error: {e}")
    finally:
        logger.info("Server stopped")

if __name__ == "__main__":
    # Create offload directory if needed
    if not os.path.exists("./offload"):
        os.makedirs("./offload")
    
    # Start the server
    serve()