# Attempt 1 report (EPISODE_LEN=50, TARGET=0.99)

- train_acc: 0.9994
- val_acc:   1.0000
- n_train:   1639
- n_val:     409
- val confusion (tp/tn/fp/fn): {'tp': 259, 'tn': 150, 'fp': 0, 'fn': 0}
- per-scenario val accuracy: {'SP': 1.0, 'SPADV_adv@0': 1.0, 'SPADV_adv@1': 1.0, 'SPADV_adv@2': 1.0}
- config: {'batch_size': 64, 'embed_dim': 256, 'hidden_dim': 256, 'num_layers': 1, 'dropout': 0.1, 'bidirectional': False, 'lr': 0.001, 'weight_decay': 1e-05, 'epochs': 30, 'early_stop_patience': 6}