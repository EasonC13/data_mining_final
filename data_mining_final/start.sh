source ~/anaconda3/etc/profile.d/conda.sh
conda activate tf20
uvicorn chatbot_service:app --port 13530 --workers 5 --proxy-headers