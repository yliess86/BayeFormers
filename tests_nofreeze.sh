python3 -m examples.bert_glue --epochs 5 --batch_size 8 --learning_rate 2e-5 --samples 10 --device "cuda:1" --delta 0.05 >> exp4.txt &\
python3 -m examples.bert_glue --epochs 5 --batch_size 8 --learning_rate 2e-6 --samples 10 --device "cuda:2" --delta 0.05 >> exp5.txt &\
python3 -m examples.bert_glue --epochs 5 --batch_size 8 --learning_rate 2e-5 --samples 3 --device "cuda:3" --delta 0.05 >> exp6.txt