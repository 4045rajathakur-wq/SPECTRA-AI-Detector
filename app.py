import os, uuid
from pathlib import Path
from flask import Flask, render_template, request, jsonify
from detector import ImageDetector
BASE=Path(__file__).resolve().parent
UPLOADS=BASE/'uploads'; UPLOADS.mkdir(exist_ok=True)
app=Flask(__name__); app.config['MAX_CONTENT_LENGTH']=100*1024*1024
detector=ImageDetector()
@app.get('/')
def home(): return render_template('index.html')
@app.get('/api/health')
def health(): return jsonify({'ok':True,'model':detector.model_id,'model_loaded':detector.loaded})
@app.post('/api/scan')
def scan():
    f=request.files.get('file'); typ=request.form.get('type','image')
    if not f or not f.filename: return jsonify(error='No file uploaded'),400
    p=UPLOADS/f'{uuid.uuid4().hex}_{Path(f.filename).name}'; f.save(p)
    try:
        if typ=='image': r=detector.predict_image(p)
        elif typ=='video': r=detector.predict_video(p)
        else: r=detector.predict_audio(p)
        return jsonify(r)
    except Exception as e: return jsonify(error=str(e)),500
    finally: p.unlink(missing_ok=True)
if __name__=='__main__': app.run(host='127.0.0.1',port=int(os.getenv('PORT',5000)),debug=True)
