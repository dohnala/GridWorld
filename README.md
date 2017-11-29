## Grid World

## DQN agent for find_treasure_v0
```
python3 -m experiments.dqn_find_treasure_v0 --train=1000 --eval=100 --eval_after=200
```

## DQN agent for find_treasure_v1
```
python3 -m experiments.dqn_find_treasure_v1 --train=10000 --eval=100 --eval_after=500
```

## N-step DQN agent for find_treasure_v0
```
python3 -m experiments.nstep_dqn_find_treasure_v0 --train=1000 --eval=100 --eval_after=200
```

## N-step DQN agent for find_treasure_v1
```
python3 -m experiments.nstep_dqn_find_treasure_v1 --train=10000 --eval=100 --eval_after=500
```

## Async N-step DQN agent for find_treasure_v1
```
python3 -m experiments.async_nstep_dqn_find_treasure_v1 --train=10000 --eval=100 --eval_after=2000
```