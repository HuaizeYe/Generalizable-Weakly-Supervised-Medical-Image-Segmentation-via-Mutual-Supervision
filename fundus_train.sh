python -W ignore train_weak.py --dataset fundus --domain_idxs 1,2,3 --test_domain_idx 0 --ram --rec --is_out_domain --consistency --consistency_type mse --save_path ../outdir/fundus/res0 --gpu 1,0 --lr 0.0001
python -W ignore train_weak.py --dataset fundus --domain_idxs 0,2,3 --test_domain_idx 1 --ram --rec --is_out_domain --consistency --consistency_type mse --save_path ../outdir/fundus/res1 --gpu 1,0 --lr 0.0001
python -W ignore train_weak.py --dataset fundus --domain_idxs 0,1,3 --test_domain_idx 2 --ram --rec --is_out_domain --consistency --consistency_type mse --save_path ../outdir/fundus/res2 --gpu 1,0 --lr 0.0001
python -W ignore train_weak.py --dataset fundus --domain_idxs 0,1,2 --test_domain_idx 3 --ram --rec --is_out_domain --consistency --consistency_type mse --save_path ../outdir/fundus/res3 --gpu 1,0 --lr 0.0001