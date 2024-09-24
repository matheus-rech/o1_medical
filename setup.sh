pip install -e .
pip install transformers
git clone https://github.com/yuh-zha/AlignScore
wget -P ./AlignScore/ckpt https://huggingface.co/yzha/AlignScore/resolve/main/AlignScore-large.ckpt 
pip install spacy
python3 -m spacy download en_core_web_sm
pip install mauve-text 