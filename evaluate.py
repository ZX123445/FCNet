import os
from evaluator.evaluator import evaluate_dataset
import torch
import time

"""
mkdir:
    Create a folder if "path" does not exist.
"""


def mkdir(path):
    if os.path.exists(path) == False:
        os.makedirs(path)


"""
write_doc:
    Write "content" into the file(".txt") in "path".
"""


def write_doc(path, content):
    with open(path, 'a') as file:
        file.write(content)


"""
get_time:
    Obtain the current time.
"""


def get_time():
    torch.cpu.synchronize()
    return time.time()


def evaluate(roots, doc_path, num_thread, pin):
    datasets = roots.keys()
    for dataset in datasets:
        # Evaluate predictions of "dataset".
        results = evaluate_dataset(roots=roots[dataset],
                                   dataset=dataset,
                                   batch_size=1,
                                   num_thread=num_thread,
                                   demical=True,
                                   suffixes={'gt': '.png', 'pred': '.png'},
                                   pin=pin)

        # Save evaluation results.
        content = '{}:\n'.format(dataset)
        # content += 'max-Fmeasure={}'.format(results['max_f'])
        content += 'max-Fmeasure={} mean-Fmeasure={} '.format(results['max_f'], results['mean_f'])
        content += 'max-Emeasure={} mean-Emeasure={} '.format(results['max_e'], results['mean_e'])
        content += ' '
        content += 'Smeasure={}'.format(results['s'])
        content += ' '
        content += 'MAE={}\n'.format(results['mae'])
        write_doc(doc_path, content)
    content = '\n'
    write_doc(doc_path, content)


eval_device = '0'
eval_doc_path = './evaluation.txt'
eval_num_thread = 4

# An example to build "eval_roots".
eval_roots = dict()
datasets = ['test']

for dataset in datasets:
    roots = {'gt': './datasets/{}/GT/'.format(dataset),
             'pred': './pred/CoCA/{}/'.format(dataset)}
    eval_roots[dataset] = roots
# ------------- end -------------

if __name__ == "__main__":
    os.environ['CUDA_VISIBLE_DEVICES'] = eval_device
    evaluate(roots=eval_roots,
             doc_path=eval_doc_path,
             num_thread=eval_num_thread,
             pin=False)
