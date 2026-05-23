
FROM python:3.11-slim

RUN apt-get update && apt-get install -y wget

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

RUN wget -O /app/model.pkl "https://huggingface.co/spaces/Bass2332/sensante/resolve/main/model.pkl?download=true"
RUN wget -O /app/encoder_sexe.pkl "https://huggingface.co/spaces/Bass2332/sensante/resolve/main/encoder_sexe.pkl?download=true"
RUN wget -O /app/encoder_region.pkl "https://huggingface.co/spaces/Bass2332/sensante/resolve/main/encoder_region.pkl?download=true"
RUN wget -O /app/feature_cols.pkl "https://huggingface.co/spaces/Bass2332/sensante/resolve/main/feature_cols.pkl?download=true"

EXPOSE 7860

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "7860"]
