FROM python:3.11-slim

ARG PORT=7860
ENV PORT=${PORT} \
	GRADIO_SERVER_NAME=0.0.0.0 \
	GRADIO_SERVER_PORT=${PORT}

WORKDIR /app

COPY requirements.txt /app/requirements.txt
RUN pip install --upgrade pip && \
	pip install -r requirements.txt

COPY app.py /app/app.py

RUN useradd -m appuser && chown -R appuser /app
USER appuser

EXPOSE ${PORT}

# Run the Gradio app
CMD ["python", "app.py"]

