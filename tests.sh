############################################################
# BERT fine-tuned - no ancestors test
echo "Running BERT fine-tuned - no ancestors test"
python main.py --no_ancestors
mv results/run_log_* results/test_bert_no_ancestors.json

# BERT fine-tuned - MAX_ANCESTORS empty comments
echo "Running BERT fine-tuned - MAX_ANCESTORS empty comments"
python main.py --used_ancestors 0
mv results/run_log_* results/test_bert_empty_comments.json

############################################################
# SARCBERT with 1, 2, 3 ancestors - from the start
echo "Running SARCBERT with 1, 2, 3 ancestors - from the start"
python main.py --used_ancestors 1 --max_ancestors 1
mv results/run_log_* results/test_start_ancestors_1.json

python main.py --used_ancestors 2 --max_ancestors 2
mv results/run_log_* results/test_start_ancestors_2.json

python main.py --used_ancestors 3 --max_ancestors 3
mv results/run_log_* results/test_start_ancestors_3.json

############################################################
# SARCBERT with 1, 2, 3 ancestors - from the end
echo "Running SARCBERT with 1, 2, 3 ancestors - from the end"
python main.py --used_ancestors 1 --max_ancestors 1 --direction_end
mv results/run_log_* results/test_end_ancestors_1.json

python main.py --used_ancestors 2 --max_ancestors 2 --direction_end
mv results/run_log_* results/test_end_ancestors_2.json

python main.py --used_ancestors 3 --max_ancestors 3 --direction_end
mv results/run_log_* results/test_end_ancestors_3.json

############################################################
# SARCBERT with unbalanced
# python main.py --train_file "data/train-pol-unbalanced.json" --test_file "data/test-pol-unbalanced.json"
# mv results/run_log_* results/test_start_ancestors_1.json

############################################################
# SARCBERT with RoBERTa
echo "SARCBERT with RoBERTa"
python main.py --roberta --batch_size=8
mv results/run_log_* results/test_roberta.json

############################################################
# SARCBERT with Electra
echo "SARCBERT with Electra"
python main.py --electra
mv results/run_log_* results/test_electra.json
