import pyttsx3
import threading
import queue
import time

class TTSEngine:
    _instance = None
    _lock = threading.Lock()

    def __new__(cls, *args, **kwargs):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super(TTSEngine, cls).__new__(cls)
                cls._instance._initialized = False
            return cls._instance

    def __init__(self, cooldown: float = 3.0):
        if self._initialized:
            return
            
        self.cooldown = cooldown
        self.speech_queue = queue.Queue()
        self.last_spoken: dict[str, float] = {}
        self.running = True
        
        self.thread = threading.Thread(target=self._worker, daemon=True)
        self.thread.start()
        
        self._initialized = True

    def _worker(self):
        engine = pyttsx3.init()
        engine.setProperty('rate', 150)
        
        while self.running:
            try:
                text = self.speech_queue.get(timeout=0.1)
                if text is None:
                    continue
                engine.say(text)
                engine.runAndWait()
                self.speech_queue.task_done()
            except queue.Empty:
                continue
            except Exception as e:
                print(f"TTS Error: {e}")

    def speak(self, text: str):
        now = time.time()
        last_time = self.last_spoken.get(text, 0)
        
        if (now - last_time) >= self.cooldown:
            self.last_spoken[text] = now
            self.speech_queue.put(text)

    def shutdown(self):
        self.running = False
        if self.thread.is_alive():
            self.thread.join(timeout=1.0)
