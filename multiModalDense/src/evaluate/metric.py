from evaluate import ANETcaptions

def calculateMetrics(reference_paths, submission_path, tIoUs):
    metrics = {}
    evaluator = ANETcaptions(reference_paths, submission_path, tIoUs)
    evaluator.evaluate()
    
    for i, tiou in enumerate(tIoUs):
        metrics[tiou] = {}

        for metric in evaluator.scores:
            score = evaluator.scores[metric][i]
            metrics[tiou][metric] = score

    # Print the averages
    
    metrics['Average across tIoUs'] = {}
    for metric in evaluator.scores:
        score = evaluator.scores[metric]
        metrics['Average across tIoUs'][metric] = sum(score) / float(len(score))
    
    return metrics