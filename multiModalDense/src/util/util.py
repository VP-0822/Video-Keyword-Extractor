
def averageOfAllMetrics(val1_dict, val2_dict):
    """
        Averages metric values from 2 validation sets and returns in new dictionary
    """
    average_dict = {}
    for metric_name in val1_dict.keys():
        val_1_metric_value = val1_dict[metric_name]
        val_2_metric_value = val1_dict[metric_name]
        average_dict[metric_name] = (val_1_metric_value + val_2_metric_value) / 2
    
    return average_dict
