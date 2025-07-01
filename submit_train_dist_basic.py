from __future__ import absolute_import, division, print_function, unicode_literals

import pykrylov
import os
import argparse
from argparse import Namespace
import yaml

ROOT_PATH = os.path.abspath(os.path.join(__file__, os.pardir))


def get_task_cmd(main_file, model_config, data_config, train_config):
    task_folder = os.path.join(os.environ['KRYLOV_WF_HOME'], 'src', os.environ['KRYLOV_WF_TASK_NAME'])
    calling_file = os.path.join(task_folder, main_file)  # 主文件
    training_conf = os.path.join(task_folder, 'config', train_config)
    cmd = f'{calling_file} --model_config {model_config} --data_config {data_config} --train_config {training_conf}'
    os.system(cmd)


def get_task(main_file, model_config: str, data_config: str, train_config: str, cluster):
    this_task = pykrylov.Task(get_task_cmd, [main_file, model_config, data_config, train_config],
                              docker_image='hub.tess.io/quccoe/qu_dev_images:xiaohuli_mm_v2')

    this_task.add_file(f'./{main_file}')
    this_task.add_package('llamavid')
    this_task.add_directory('configs')
    this_task.add_cpu(12)
    this_task.add_memory(450)
    # this_task.run_on_gpu(4, 'a100')
    this_task.add_execution_parameter('accelerator', {'type': 'gpu', 'quantity': '4'})
    this_task.mount_pvc('yinilin', 'krylov-user-pvc-yinilin', cluster=cluster)  ###这个咋改？
    return this_task


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Query Item Kry Model')

    # parser.add_argument('--namespace', default='search-preemptive-a100', type=str)
    parser.add_argument('--namespace', default='ebay', type=str)
    parser.add_argument('--cluster', default='tess94', type=str)
    parser.add_argument('--main_file', default='train_mem.py', type=str)
    parser.add_argument('--model_config', default='model_config.yml', type=str)
    parser.add_argument('--data_config', default='data_config.yml', type=str)
    parser.add_argument('--train_config', default='train_config.yml', type=str)

    args = parser.parse_args()

    model_config_path = os.path.join(ROOT_PATH, args.model_config)
    data_config_path = os.path.join(ROOT_PATH, args.data_config)
    train_config_path = os.path.join(ROOT_PATH, args.train_config)

    task = get_task(args.main_file, args.model_config, args.data_config, args.train_config, args.cluster)
    workflow = pykrylov.Flow(task)
    workflow.execution_parameters.add_execution_parameter("enableChooseCluster", "true")

    project_name = 'test010'
    job_name = 'llamavid_train'
    experiment_name = 'v1'

    session = pykrylov.Session(namespace=args.namespace, project_name=project_name, job_name=job_name,
                               do_check_update=False)
    exp_id = session.submit_experiment(workflow, project=project_name, experiment_name=experiment_name)

    run_id = pykrylov.ems.show_experiment(exp_id).get("runtime").get("workflow").get("runId")
    print(f"Experiment {experiment_name} has been submitted Krylov with exp id: {exp_id}, run id: {run_id}")
    print(session.get_work_directory())
