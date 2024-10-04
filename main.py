from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from TTS.api import TTS
import os
from pydantic import BaseModel
import uvicorn

app = FastAPI()

# Initialize TTS model
tts = TTS(model_name="tts_models/multilingual/multi-dataset/xtts_v2", progress_bar=False)

# Directory to store reference audio files
REFERENCE_AUDIO_DIR = "cloning"

class TTSRequest(BaseModel):
    text: str
    speaker: str

@app.post("/tts")
async def text_to_speech(request: TTSRequest):
    reference_audio_path = os.path.join(REFERENCE_AUDIO_DIR, f"{request.speaker}.wav")

    if not os.path.exists(reference_audio_path):
        raise HTTPException(status_code=404, detail=f"No reference audio found for {request.speaker}")

    # Generate speech
    out_path = f"temp_{request.speaker}.wav"
    tts.tts_to_file(
        text=request.text,
        file_path=out_path,
        speaker_wav=reference_audio_path,
        language="ru"
    )

    # Send audio file
    return FileResponse(out_path, media_type="audio/wav")

if __name__ == '__main__':
    os.makedirs(REFERENCE_AUDIO_DIR, exist_ok=True)
    uvicorn.run(app, host="0.0.0.0", port=5000)