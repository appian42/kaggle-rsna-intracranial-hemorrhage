gpu=0
tta=5

predict_valid() {
    model=$1
    fold=$2
    ep=$3

    conf=./conf/${model}.py
    snapshot=./model/${model}/fold${fold}_ep${ep}.pt
    valid=./model/${model}/fold${fold}_ep${ep}_valid_tta${tta}.pkl

    python -m src.cnn.main valid ${conf} --snapshot ${snapshot} --output ${valid} --n-tta ${tta} --fold ${fold} --gpu ${gpu}
}

predict_test() {
    model=$1
    fold=$2
    ep=$3

    conf=./conf/${model}.py
    snapshot=./model/${model}/fold${fold}_ep${ep}.pt
    test=./model/${model}/fold${fold}_ep${ep}_test_tta${tta}.pkl

    python -m src.cnn.main test ${conf} --snapshot ${snapshot} --output ${test} --n-tta ${tta} --fold ${fold} --gpu ${gpu}
}


predict_valid model100 0 2
predict_valid model100 0 3
predict_valid model100 1 2
predict_valid model100 1 3
predict_valid model100 2 2
predict_valid model100 2 3
predict_valid model100 3 2
predict_valid model100 3 3
predict_valid model100 4 2
predict_valid model100 4 3
predict_valid model100 5 2
predict_valid model100 5 3
predict_valid model100 6 2
predict_valid model100 6 3
predict_valid model100 7 2
predict_valid model100 7 3

predict_valid model110 0 2
predict_valid model110 0 3
predict_valid model110 1 2
predict_valid model110 1 3
predict_valid model110 2 2
predict_valid model110 2 3
predict_valid model110 3 2
predict_valid model110 3 3
predict_valid model110 4 2
predict_valid model110 4 3
predict_valid model110 5 2
predict_valid model110 5 3
predict_valid model110 6 2
predict_valid model110 6 3
predict_valid model110 7 2
predict_valid model110 7 3


predict_test model100 0 2
predict_test model100 0 3
predict_test model100 1 2
predict_test model100 1 3
predict_test model100 2 2
predict_test model100 2 3
predict_test model100 3 2
predict_test model100 3 3
predict_test model100 4 2
predict_test model100 4 3
predict_test model100 5 2
predict_test model100 5 3
predict_test model100 6 2
predict_test model100 6 3
predict_test model100 7 2
predict_test model100 7 3

predict_test model110 0 2
predict_test model110 0 3
predict_test model110 1 2
predict_test model110 1 3
predict_test model110 2 2
predict_test model110 2 3
predict_test model110 3 2
predict_test model110 3 3
predict_test model110 4 2
predict_test model110 4 3
predict_test model110 5 2
predict_test model110 5 3
predict_test model110 6 2
predict_test model110 6 3
predict_test model110 7 2
predict_test model110 7 3
