from aif360 import datasets as ds

def aif360_dataset(x, y, protected_attribute_names, privileged_classes, favorable_classes):
    df = x.join(y)
    aif_ds = ds.StandardDataset(df,
                                label_name='y',  # 'string'
                                favorable_classes=favorable_classes,  # array [1]
                                protected_attribute_names=protected_attribute_names,  # array ['race']
                                privileged_classes=privileged_classes  # array of array [[1]]
                                )
    return aif_ds

def get_aif_dataset(x, y, label, protected_attribute_names, privileged_classes, favorable_classes):
    df = x.join(y)
    aif_ds = ds.StandardDataset(df,
                                label_name=label,                                      # 'string'
                                favorable_classes=favorable_classes,                   # array [1]
                                protected_attribute_names=protected_attribute_names,   # array ['race']
                                privileged_classes=privileged_classes                  # array of array [[1]]
                                )

    return aif_ds


def get_aif_regression_dataset(x, y, label, protected_attribute_names, privileged_classes, favorable_classes):
    df = x.join(y)
    aif_ds = ds.RegressionDataset(df,
                                  dep_var_name=label,  # 'string'
                                  # favorable_classes=favorable_classes,                   # array [1]
                                  protected_attribute_names=protected_attribute_names,  # array ['race']
                                  privileged_classes=privileged_classes  # array of array [[1]]
                                  )

    return aif_ds
