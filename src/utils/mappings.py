counter = {
    'all': 674258,
    'negative': 577155, # 0.8559

    'any': 97103, # 0.1440
    'epidural': 2761, # 0.0040
    'subdural': 42496, # 0.0630
    'subarachnoid': 32122, # 0.0476
    'intraventricular': 23766, # 0.0352
    'intraparenchymal': 32564, # 0.0482
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
