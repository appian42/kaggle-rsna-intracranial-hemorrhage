gpu=0

train() {
    model=$1
    fold=$2

    conf=./conf/${model}.py
    python -m src.cnn.main train ${conf} --fold ${fold} --gpu ${gpu}
}

train model100 0
train model100 1
train model100 2
train model100 3
train model100 4
train model100 5
train model100 6
train model100 7

train model110 0
train model110 1
train model110 2
train model110 3
train model110 4
train model110 5
train model110 6
train model110 7

