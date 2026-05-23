
FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

RUN pip install huggingface_hub && python -c "from huggingface_hub import hf_hub_download; import shutil; [shutil.copy(hf_hub_download(repo_id='Bass2332/sensante', filename=f, repo_type='space'), f'/app/{f}') for f in ['model.pkl','encoder_sexe.pkl','encoder_region.pkl','feature_cols.pkl']]"

EXPOSE 7860

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "7860"]
