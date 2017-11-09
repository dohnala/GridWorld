## Grid World

## DQN agent for find_treasure_v0
### Train model
```
python3 -m experiments.dqn_find_treasure_v0 --train=300 --eval=30
```
### User pre-trained model
```
python3 -m experiments.dqn_find_treasure_v0 --eval=30 --load=trained/dqn_find_treasure_v0.ckp
```

## DQN agent for find_treasure_v1
### Train model
```
python3 -m experiments.dqn_find_treasure_v1 --train=3000 --eval=100 --log_every=100
```
### User pre-trained model
```
python3 -m experiments.dqn_find_treasure_v1 --eval=100 --load=trained/dqn_find_treasure_v1.ckp
```