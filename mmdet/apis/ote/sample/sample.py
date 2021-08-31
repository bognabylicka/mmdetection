# Copyright (C) 2021 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions
# and limitations under the License.

import argparse
import logging
import os.path as osp
import sys

from ote_sdk.configuration.helper import create
from ote_sdk.entities.inference_parameters import InferenceParameters
from ote_sdk.entities.model import ModelStatus
from ote_sdk.entities.model_template import parse_model_template
from ote_sdk.entities.subset import Subset
from ote_sdk.entities.task_environment import TaskEnvironment
from sc_sdk.entities.model_storage import NullModelStorage
from sc_sdk.entities.optimized_model import ModelOptimizationType
from sc_sdk.entities.optimized_model import ModelPrecision
from sc_sdk.entities.optimized_model import OptimizedModel
from sc_sdk.entities.optimized_model import TargetDevice
from sc_sdk.entities.project import NullProject
from sc_sdk.usecases.tasks.interfaces.export_interface import ExportType

from mmdet.apis.ote.apis.detection.config_utils import set_values_as_default
from mmdet.apis.ote.apis.detection.ote_utils.dataset import MMDataset
from mmdet.apis.ote.apis.detection.ote_utils.label_schema import FlatLabelSchema
from mmdet.apis.ote.apis.detection.ote_utils.misc import get_task_class
from mmdet.apis.ote.apis.detection.ote_utils.misc import reload_hyper_parameters
from mmdet.apis.ote.apis.detection.ote_utils.model import Model
from mmdet.apis.ote.apis.detection.ote_utils.result_set import ResultSet

logger = logging.getLogger(__name__)



def parse_args():
    parser = argparse.ArgumentParser(description='Sample showcasing the new API')
    parser.add_argument('template_file_path', help='path to template file')
    parser.add_argument('--data-dir', default='data')
    parser.add_argument('--export', action='store_true')
    args = parser.parse_args()
    return args


def main(args):
    logger.info('Initialize dataset')
    dataset = MMDataset(
            annotation_files={
                Subset.TRAINING: osp.join(args.data_dir, 'coco/annotations/instances_val2017.json'),
                Subset.VALIDATION: osp.join(args.data_dir, 'coco/annotations/instances_val2017.json'),
                Subset.TESTING: osp.join(args.data_dir, 'coco/annotations/instances_val2017.json')
            },
            data_root_dirs={
                Subset.TRAINING: osp.join(args.data_dir, 'coco/val2017/'),
                Subset.VALIDATION: osp.join(args.data_dir, 'coco/val2017/'),
                Subset.TESTING: osp.join(args.data_dir, 'coco/val2017/')
            })

    labels = dataset.get_labels()
    labels_schema = FlatLabelSchema(labels)

    logger.info(f'Train dataset: {len(dataset.get_subset(Subset.TRAINING))} items')
    logger.info(f'Validation dataset: {len(dataset.get_subset(Subset.VALIDATION))} items')

    logger.info('Load model template')
    model_template = parse_model_template(args.template_file_path, '1')

    # Here we have to reload parameters manually because
    # `parse_model_template` was called when `configuration.yaml` was not near `template.yaml.`
    if not model_template.hyper_parameters.data:
        reload_hyper_parameters(model_template)

    hyper_parameters = model_template.hyper_parameters.data
    set_values_as_default(hyper_parameters)

    logger.info('Setup environment')
    params = create(hyper_parameters)
    logger.info('Set hyperparameters')
    params.learning_parameters.num_iters = 10
    environment = TaskEnvironment(model=None, hyper_parameters=params, label_schema=labels_schema, model_template=model_template)

    logger.info('Create base Task')
    task_impl_path = model_template.entrypoints.base
    task_cls = get_task_class(task_impl_path)
    task = task_cls(task_environment=environment)

    logger.info('Train model')
    output_model = Model(
        dataset,
        environment.get_model_configuration(),
        model_status=ModelStatus.NOT_READY
    )
    task.train(dataset, output_model)

    logger.info('Get predictions on the validation set')
    validation_dataset = dataset.get_subset(Subset.VALIDATION)
    predicted_validation_dataset = task.infer(
        validation_dataset.with_empty_annotations(),
        InferenceParameters(is_evaluation=True))
    resultset = ResultSet(
        model=output_model,
        ground_truth_dataset=validation_dataset,
        prediction_dataset=predicted_validation_dataset,
    )
    logger.info('Estimate quality on validation set')
    performance = task.evaluate(resultset)
    logger.info(str(performance))

    if args.export:
        logger.info('Export model')
        exported_model = OptimizedModel(
            NullProject(),
            NullModelStorage(),
            dataset,
            environment.get_model_configuration(),
            ModelOptimizationType.MO,
            precision=[ModelPrecision.FP32],
            optimization_methods=[],
            optimization_level={},
            target_device=TargetDevice.UNSPECIFIED,
            performance_improvement={},
            model_size_reduction=1.,
            model_status=ModelStatus.NOT_READY)
        task.export(ExportType.OPENVINO, exported_model)

        logger.info('Create OpenVINO Task')
        environment.model = exported_model
        openvino_task_impl_path = model_template.entrypoints.openvino
        openvino_task_cls = get_task_class(openvino_task_impl_path)
        openvino_task = openvino_task_cls(environment)

        logger.info('Get predictions on the validation set')
        predicted_validation_dataset = openvino_task.infer(
            validation_dataset.with_empty_annotations(),
            InferenceParameters(is_evaluation=True))
        resultset = ResultSet(
            model=output_model,
            ground_truth_dataset=validation_dataset,
            prediction_dataset=predicted_validation_dataset,
        )
        logger.info('Estimate quality on validation set')
        performance = openvino_task.evaluate(resultset)
        logger.info(str(performance))


if __name__ == '__main__':
    args = parse_args()
    sys.exit(main(args) or 0)
