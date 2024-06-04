# HyperCL

HyperCL is a universal contrastive learning framework for hyper-relational KG embeddings, which is flexible to integrate different hyper-relational KG embedding methods and effectively boost their link prediction performance with hierarchical ontology. Please see the details in our paper below:
- Yuhuan Lu, Weijian Yu, Xin Jing, and Dingqi Yang. 2024. HyperCL: A Contrastive Learning Framework for Hyper-Relational Knowledge Graph Embedding with Hierarchical Ontology, accepted by ACL Findings 2024.

## How to run the code
###### Train and evaluate model (suggested parameters for JF17k, WikiPeople and WD50K dataset)
```
python ./src/run.py --dataset "jf17k" --device "0" --vocab_size 29148 --vocab_schema_size 29393 --vocab_file "./data/jf17k/vocab.txt" --vocab_schema_file "./data/jf17k/vocab_schema.txt" --ent2types_file "./data/jf17k/entity2types_ttv.txt" --train_file "./data/jf17k/train.json" --test_file "./data/jf17k/test.json" --ground_truth_file "./data/jf17k/all.json" --num_workers 1 --num_relations 501 --num_types 748 --max_seq_len 11 --max_arity 6 --hidden_dim 256 --global_layers 2 --global_dropout 0.9 --global_activation "elu" --global_heads 4 --local_layers 12 --local_dropout 0.35 --local_heads 4 --decoder_activation "gelu" --batch_size 1024 --cl_batch_size 8192 --lr 5e-4 --cl_lr 5e-4 --weight_deca 0.002 --entity_soft 0.9 --relation_soft 0.9 --hyperedge_dropout 0.85 --epoch 300 --warmup_proportion 0.05

python ./src/run.py --dataset "wikipeople" --device "2" --vocab_size 35005 --vocab_schema_size 38221 --vocab_file "./data/wikipeople/vocab.txt" --vocab_schema_file "./data/wikipeople/vocab_schema.txt" --ent2types_file "./data/wikipeople/entity2types_ttv.txt" --train_file "./data/wikipeople/train+valid.json" --test_file "./data/wikipeople/test.json" --ground_truth_file "./data/wikipeople/all.json" --num_workers 1 --num_relations 178 --num_types 3396 --max_seq_len 13 --max_arity 7 --hidden_dim 256 --global_layers 2 --global_dropout 0.1 --global_activation "elu" --global_heads 4 --local_layers 12 --local_dropout 0.1 --local_heads 4 --decoder_activation "gelu" --batch_size 1024 --cl_batch_size 12288 --lr 5e-4 --cl_lr 5e-4 --weight_deca 0.01 --entity_soft 0.2 --relation_soft 0.1 --hyperedge_dropout 0.99 --epoch 300 --warmup_proportion 0.1

python ./src/run.py --dataset "wd50k" --device "3" --vocab_size 47688 --vocab_schema_size 53475 --vocab_file "./data/wd50k/vocab.txt" --vocab_schema_file "./data/wd50k/vocab_schema.txt" --ent2types_file "./data/wd50k/entity2types_ttv.txt" --train_file "./data/wd50k/train+valid.json" --test_file "./data/wd50k/test.json" --ground_truth_file "./data/wd50k/all.json" --num_workers 1 --num_relations 531 --num_types 6320 --max_seq_len 19 --max_arity 10 --hidden_dim 256 --global_layers 2 --global_dropout 0.1 --global_activation "elu" --global_heads 4 --local_layers 12 --local_dropout 0.1 --local_heads 4 --decoder_activation "gelu" --batch_size 512 --cl_batch_size 8192 --lr 5e-4 --cl_lr 1e-3 --weight_deca 0.01 --entity_soft 0.2 --relation_soft 0.1 --hyperedge_dropout 0.8 --epoch 300 --warmup_proportion 0.1
```

# Python lib versions
This project should work fine with the following environments:
- Python 3.9.16 for data preprocessing, training and evaluation with:
    -  torch 1.10.0
    -  torch-scatter 2.0.9
    -  torch-sparse 0.6.13
    -  torch-cluster 1.6.0
    -  torch-geometric 2.1.0.post1
    -  numpy 1.23.3
- GPU with CUDA 11.3

# Reference
If you use our code or datasets, please cite:
```
@inproceedings{lu2024hypercl,
  title={HyperCL: A Contrastive Learning Framework for Hyper-Relational Knowledge Graph Embedding with Hierarchical Ontology},
  author={Lu, Yuhuan and Yu, Weijian and Jing, Xin and Yang, Dingqi},
  booktitle={xxx},
  pages={xxx},
  year={2024}
}
```
