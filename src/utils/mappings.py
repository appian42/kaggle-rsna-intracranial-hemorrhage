counter = {
    'all': 674258,
    'negative': 577155,

    'any': 97103,
    'epidural': 2761,
    'subdural': 42496,
    'subarachnoid': 32122,
    'intraventricular': 23766,
    'intraparenchymal': 32564,
}


label_to_num = {
    'any': 0,
    'epidural': 1,
    'subdural': 2,
    'subarachnoid': 3,
    'intraventricular': 4,
    'intraparenchymal': 5,
}
num_to_label = {v:k for k,v in label_to_num.items()}
