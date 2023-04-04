import sentencepiece

# tokenizer:
# spm_train --input=data/all.txt --model_prefix=code-bpe-2k --vocab_size=2048 --character_coverage=0.9995 --model_type=bpe --input_sentence_size=1000000 --bos_id=-1 --eos_id=-1 --remove_extra_whitespaces=false --allow_whitespace_only_pieces=true --normalization_rule_name=identity
spm = sentencepiece.SentencePieceProcessor(model_file="code-bpe-2k.model")

vocab_size = 2048
encode = spm.encode
decode = spm.decode