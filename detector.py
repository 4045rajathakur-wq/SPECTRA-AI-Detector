import os, subprocess, tempfile
from pathlib import Path
from PIL import Image
import numpy as np
MODEL_ID=os.getenv('SPECTRA_MODEL','umm-maybe/AI-image-detector')
class ImageDetector:
 def __init__(self):
  self.model_id=MODEL_ID; self.pipe=None; self.loaded=False
  try:
   from transformers import pipeline
   import torch
   self.pipe=pipeline('image-classification',model=self.model_id,device=0 if torch.cuda.is_available() else -1); self.loaded=True
  except Exception as e: self.load_error=str(e)
 def _score(self,rs):
  aiw=('ai','artificial','fake','generated','synthetic','deepfake'); rw=('real','human','authentic','natural','original'); ai=[]; real=[]
  for x in rs:
   l=str(x.get('label','')).lower(); s=float(x.get('score',0))
   if any(w in l for w in aiw): ai.append(s)
   elif any(w in l for w in rw): real.append(s)
  if ai and real: return max(ai),max(real)
  if ai: return max(ai),1-max(ai)
  if real: return 1-max(real),max(real)
  raise RuntimeError('Model labels could not be mapped to AI/real. Set SPECTRA_MODEL to a compatible model.')
 def predict_image(self,p):
  if not self.loaded: raise RuntimeError('AI model failed to load: '+self.load_error)
  im=Image.open(p).convert('RGB'); rs=self.pipe(im,top_k=5); ai,real=self._score(rs)
  return {'media_type':'image','verdict':'AI-GENERATED' if ai>=.5 else 'LIKELY REAL','ai_probability':round(ai*100,2),'real_probability':round(real*100,2),'confidence':round(max(ai,real)*100,2),'signals':[{'name':'AI classifier','value':round(ai*100,1),'level':'High' if ai>=.75 else 'Medium'},{'name':'Real classifier','value':round(real*100,1),'level':'High' if real>=.75 else 'Medium'},{'name':'Resolution','value':f'{im.width} × {im.height}','level':'Info'},{'name':'Format','value':im.format or 'Unknown','level':'Info'}],'model':self.model_id}
 def predict_video(self,p):
  if not self.loaded: raise RuntimeError('AI model failed to load: '+self.load_error)
  with tempfile.TemporaryDirectory() as td:
   out=Path(td)/'f_%03d.jpg'; q=subprocess.run(['ffmpeg','-hide_banner','-loglevel','error','-i',str(p),'-vf','fps=1/2,scale=768:-1','-frames:v','8',str(out)],capture_output=True,text=True)
   if q.returncode: raise RuntimeError('FFmpeg error: '+q.stderr[-400:])
   fs=sorted(Path(td).glob('f_*.jpg'))
   if not fs: raise RuntimeError('No frames could be extracted.')
   vals=[self.predict_image(x)['ai_probability'] for x in fs]; ai=float(np.mean(vals)); real=100-ai
   return {'media_type':'video','verdict':'AI-GENERATED' if ai>=50 else 'LIKELY REAL','ai_probability':round(ai,2),'real_probability':round(real,2),'confidence':round(max(ai,real),2),'signals':[{'name':'Frames sampled','value':len(fs),'level':'Info'},{'name':'Average AI probability','value':round(ai,1),'level':'High' if ai>=75 else 'Medium'}],'model':self.model_id}
 def predict_audio(self,p):
  raise RuntimeError('Audio detection needs a dedicated speech/audio deepfake model. This build will not invent an audio verdict.')
