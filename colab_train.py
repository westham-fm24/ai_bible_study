!git clone https://github.com/westham-fm24/ai_bible_study.git
%cd ai_bible_study

!pip install torch transformers tokenizers datasets scikit-learn tqdm pyyaml

!python scripts/preprocess.py
!python scripts/train_tokenizer.py
!python scripts/train.py

!test -f scripts/infer.py && python scripts/infer.py || echo "No infer.py script found."
!ls models/