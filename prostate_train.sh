python -W ignore train_weak.py --dataset prostate --domain_idxs 1,2,3,4,5 --test_domain_idx 0 --ram --rec --is_out_domain --consistency --consistency_type mse --save_path ../outdir/prostate/res0 --gpu 2,3 --lr 0.0001
python -W ignore train_weak.py --dataset prostate --domain_idxs 0,2,3,4,5 --test_domain_idx 1 --ram --rec --is_out_domain --consistency --consistency_type mse --save_path ../outdir/prostate/res1 --gpu 2,3 --lr 0.0001
python -W ignore train_weak.py --dataset prostate --domain_idxs 0,1,3,4,5 --test_domain_idx 2 --ram --rec --is_out_domain --consistency --consistency_type mse --save_path ../outdir/prostate/res2 --gpu 2,3 --lr 0.0001
python -W ignore train_weak.py --dataset prostate --domain_idxs 0,1,2,4,5 --test_domain_idx 3 --ram --rec --is_out_domain --consistency --consistency_type mse --save_path ../outdir/prostate/res3 --gpu 2,3 --lr 0.0001
python -W ignore train_weak.py --dataset prostate --domain_idxs 0,1,2,3,5 --test_domain_idx 4 --ram --rec --is_out_domain --consistency --consistency_type mse --save_path ../outdir/prostate/res4 --gpu 2,3 --lr 0.0001
python -W ignore train_weak.py --dataset prostate --domain_idxs 0,1,2,3,4 --test_domain_idx 5 --ram --rec --is_out_domain --consistency --consistency_type mse --save_path ../outdir/prostate/res5 --gpu 2,3 --lr 0.0001