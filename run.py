import multiprocessing
import timeit
import traceback

import yaml
from dotmap import DotMap

from model import Model, Request

with open('requests.yaml') as f:
    data = yaml.safe_load(f)
    settings = DotMap(data, _dynamic=False)


def execute_request(evaluation_request, iteration):
    print(f'[{evaluation_request.name}][{iteration}] Executing a request')
    try:
        evaluation_request.repetition_iteration = iteration
        evaluation_request.split_seed = iteration
        evaluation_request.train_seed = iteration + 1
        m = Model(Request(**evaluation_request))
        m.try_load_or_compute_input_data().train().save_results()
    except Exception as e:
        print(f'[{evaluation_request.name}] Error while executing request: {e}')
        traceback.print_exc()


for request in settings.requests:
    num_repetitions = request.run_num_repetitions \
        if request.run_num_repetitions is not None \
        else 1
    del request.run_num_repetitions

    for rep_it in range(num_repetitions):
        start_time = timeit.default_timer()

        dynamic_request = {'repetition_iteration': None, 'split_seed': None, 'train_seed': None}
        evaluation_request = DotMap({**settings.default} | {**request} | dynamic_request, _dynamic=False)
        process_eval = multiprocessing.Process(
            target=execute_request,
            args=(evaluation_request, rep_it)
        )
        process_eval.start()
        process_eval.join()

        elapsed_time = timeit.default_timer() - start_time
        print(f'[{request.name}] Request executed in {elapsed_time} s')
