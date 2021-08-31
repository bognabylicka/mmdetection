import io
import json
import logging
import os
import os.path as osp
import random
import time
import unittest
import warnings
from concurrent.futures import ThreadPoolExecutor
from subprocess import run
from typing import List
from typing import Optional
from typing import Tuple

import numpy as np
import torch
import yaml
from ote_sdk.configuration.helper import convert
from ote_sdk.configuration.helper import create
from ote_sdk.entities.annotation import Annotation
from ote_sdk.entities.datasets import Subset
from ote_sdk.entities.id import ID
from ote_sdk.entities.label import ScoredLabel
from ote_sdk.entities.metrics import Performance
from ote_sdk.entities.model import ModelStatus
from ote_sdk.entities.model_template import parse_model_template
from ote_sdk.entities.shapes.box import Box
from ote_sdk.entities.shapes.ellipse import Ellipse
from ote_sdk.entities.shapes.polygon import Point
from ote_sdk.entities.shapes.polygon import Polygon
from ote_sdk.entities.task_environment import TaskEnvironment
from ote_sdk.entities.train_parameters import TrainParameters
from sc_sdk.entities.model import NullModelStorage
from sc_sdk.entities.optimized_model import ModelOptimizationType
from sc_sdk.entities.optimized_model import ModelPrecision
from sc_sdk.entities.optimized_model import OptimizedModel
from sc_sdk.entities.optimized_model import TargetDevice
from sc_sdk.usecases.tasks.interfaces.export_interface import ExportType
from sc_sdk.usecases.tasks.interfaces.export_interface import IExportTask
from sc_sdk.utils.project_factory import NullProject

from mmdet.apis.ote.apis.detection import OpenVINODetectionTask
from mmdet.apis.ote.apis.detection import OTEDetectionConfig
from mmdet.apis.ote.apis.detection import OTEDetectionTask
from mmdet.apis.ote.apis.detection.config_utils import set_values_as_default
from mmdet.apis.ote.apis.detection.ote_utils.dataset import MMDataset
from mmdet.apis.ote.apis.detection.ote_utils.dataset_item import MMDatasetItem
from mmdet.apis.ote.apis.detection.ote_utils.image import Image
from mmdet.apis.ote.apis.detection.ote_utils.label import Label
from mmdet.apis.ote.apis.detection.ote_utils.label_schema import FlatLabelSchema
from mmdet.apis.ote.apis.detection.ote_utils.misc import reload_hyper_parameters
from mmdet.apis.ote.apis.detection.ote_utils.model import Model
from mmdet.apis.ote.apis.detection.ote_utils.result_set import ResultSet

logger = logging.getLogger(__name__)


class ModelTemplateTestCase(unittest.TestCase):

    def test_reading_mnv2_ssd_256(self):
        parse_model_template('./configs/ote/custom-object-detection/mobilenet_v2-2s_ssd-256x256/template.yaml', '1')

    def test_reading_mnv2_ssd_384(self):
        parse_model_template('./configs/ote/custom-object-detection/mobilenet_v2-2s_ssd-384x384/template.yaml', '1')

    def test_reading_mnv2_ssd_512(self):
        parse_model_template('./configs/ote/custom-object-detection/mobilenet_v2-2s_ssd-512x512/template.yaml', '1')

    def test_reading_mnv2_ssd(self):
        parse_model_template('./configs/ote/custom-object-detection/mobilenetV2_SSD/template.yaml', '1')

    def test_reading_mnv2_atss(self):
        parse_model_template('./configs/ote/custom-object-detection/mobilenetV2_ATSS/template.yaml', '1')

    def test_reading_resnet50_vfnet(self):
        parse_model_template('./configs/ote/custom-object-detection/resnet50_VFNet/template.yaml', '1')


def test_configuration_yaml():
    configuration = OTEDetectionConfig(workspace_id=ID(), model_storage_id=ID())
    configuration_yaml_str = convert(configuration, str)
    configuration_yaml_converted = yaml.safe_load(configuration_yaml_str)
    with open(osp.join('mmdet', 'apis', 'ote', 'apis', 'detection', 'configuration.yaml')) as read_file:
        configuration_yaml_loaded = yaml.safe_load(read_file)
    del configuration_yaml_converted['algo_backend']
    assert configuration_yaml_converted == configuration_yaml_loaded


def test_set_values_as_default():
    template_dir = './configs/ote/custom-object-detection/mobilenet_v2-2s_ssd-256x256/'
    template_file = osp.join(template_dir, 'template.yaml')
    model_template = parse_model_template(template_file, '1')

    # Here we have to reload parameters manually because
    # `parse_model_template` was called when `configuration.yaml` was not near `template.yaml.`
    if not model_template.hyper_parameters.data:
        reload_hyper_parameters(model_template)

    hyper_parameters = model_template.hyper_parameters.data
    # value that comes from template.yaml
    default_value = hyper_parameters['learning_parameters']['batch_size']['default_value']
    # value that comes from OTEDetectionConfig
    value = hyper_parameters['learning_parameters']['batch_size']['value']
    assert value == 5
    assert default_value == 64

    # after this call value must be equal to default_value
    set_values_as_default(hyper_parameters)
    value = hyper_parameters['learning_parameters']['batch_size']['value']
    assert default_value == value
    hyper_parameters = create(hyper_parameters)
    assert default_value == hyper_parameters.learning_parameters.batch_size


class SampleTestCase(unittest.TestCase):
    root_dir = '/tmp'
    coco_dir = osp.join(root_dir, 'data/coco')
    snapshots_dir = osp.join(root_dir, 'snapshots')

    custom_operations = ['ExperimentalDetectronROIFeatureExtractor',
                         'PriorBox', 'PriorBoxClustered', 'DetectionOutput',
                         'DeformableConv2D']

    @staticmethod
    def shorten_annotation(src_path, dst_path, num_images):
        with open(src_path) as read_file:
            content = json.load(read_file)
            selected_indexes = sorted([item['id'] for item in content['images']])
            selected_indexes = selected_indexes[:num_images]
            content['images'] = [item for item in content['images'] if
                                 item['id'] in selected_indexes]
            content['annotations'] = [item for item in content['annotations'] if
                                      item['image_id'] in selected_indexes]
            content['licenses'] = [item for item in content['licenses'] if
                                   item['id'] in selected_indexes]

        with open(dst_path, 'w') as write_file:
            json.dump(content, write_file)

    @classmethod
    def setUpClass(cls):
        cls.test_on_full = False
        os.makedirs(cls.coco_dir, exist_ok=True)
        if not osp.exists(osp.join(cls.coco_dir, 'val2017.zip')):
            run(f'wget --no-verbose http://images.cocodataset.org/zips/val2017.zip -P {cls.coco_dir}',
            check=True, shell=True)
        if not osp.exists(osp.join(cls.coco_dir, 'val2017')):
            run(f'unzip {osp.join(cls.coco_dir, "val2017.zip")} -d {cls.coco_dir}', check=True, shell=True)
        if not osp.exists(osp.join(cls.coco_dir, "annotations_trainval2017.zip")):
            run(f'wget --no-verbose http://images.cocodataset.org/annotations/annotations_trainval2017.zip -P {cls.coco_dir}',
            check=True, shell=True)
        if not osp.exists(osp.join(cls.coco_dir, 'annotations/instances_val2017.json')):
            run(f'unzip -o {osp.join(cls.coco_dir, "annotations_trainval2017.zip")} -d {cls.coco_dir}',
            check=True, shell=True)

        if cls.test_on_full:
            cls.shorten_to = 5000
        else:
            cls.shorten_to = 100

        cls.shorten_annotation(osp.join(cls.coco_dir, 'annotations/instances_val2017.json'),
                               osp.join(cls.coco_dir, 'annotations/instances_val2017.json'),
                               cls.shorten_to)

    def test_sample_on_cpu(self):
        output = run('export CUDA_VISIBLE_DEVICES=;'
                     'python mmdet/apis/ote/sample/sample.py '
                     f'--data-dir {self.coco_dir}/.. '
                     '--export configs/ote/custom-object-detection/mobilenet_v2-2s_ssd-256x256/template.yaml',
                     shell=True, check=True)
        assert output.returncode == 0

    def test_sample_on_gpu(self):
        output = run('export CUDA_VISIBLE_DEVICES=0;'
                     'python mmdet/apis/ote/sample/sample.py '
                     f'--data-dir {self.coco_dir}/.. '
                     '--export configs/ote/custom-object-detection/mobilenet_v2-2s_ssd-256x256/template.yaml',
                     shell=True, check=True)
        assert output.returncode == 0


class APITestCase(unittest.TestCase):
    """
    Collection of tests for OTE API and OTE Model Templates
    """

    @staticmethod
    def generate_random_annotated_image(
        image_width: int,
        image_height: int,
        labels: List[Label],
        min_size=50,
        max_size=250,
        shape: Optional[str] = None,
        max_shapes: int = 10,
        intensity_range: List[Tuple[int, int]] = None,
        random_seed: Optional[int] = None,
    ) -> Tuple[np.ndarray, List[Annotation]]:
        """
        Generate a random image with the corresponding annotation entities.

        :param intensity_range: Intensity range for RGB channels ((r_min, r_max), (g_min, g_max), (b_min, b_max))
        :param max_shapes: Maximum amount of shapes in the image
        :param shape: {"rectangle", "ellipse", "triangle"}
        :param image_height: Height of the image
        :param image_width: Width of the image
        :param labels: Task Labels that should be applied to the respective shape
        :param min_size: Minimum size of the shape(s)
        :param max_size: Maximum size of the shape(s)
        :param random_seed: Seed to initialize the random number generator
        :return: uint8 array, list of shapes
        """
        from skimage.draw import random_shapes
        from skimage.draw import rectangle

        if intensity_range is None:
            intensity_range = [(100, 200)]

        image1: Optional[np.ndarray] = None
        sc_labels = []
        # Sporadically, it might happen there is no shape in the image, especially on low-res images.
        # It'll retry max 5 times until we see a shape, and otherwise raise a runtime error
        if (
            shape == "ellipse"
        ):  # ellipse shape is not available in random_shapes function. use circle instead
            shape = "circle"
        for _ in range(5):
            rand_image, sc_labels = random_shapes(
                (image_height, image_width),
                min_shapes=1,
                max_shapes=max_shapes,
                intensity_range=intensity_range,
                min_size=min_size,
                max_size=max_size,
                shape=shape,
                random_seed=random_seed,
            )
            num_shapes = len(sc_labels)
            if num_shapes > 0:
                image1 = rand_image
                break

        if image1 is None:
            raise RuntimeError(
                "Was not able to generate a random image that contains any shapes"
            )

        annotations: List[Annotation] = []
        for sc_label in sc_labels:
            sc_label_name = sc_label[0]
            sc_label_shape_r = sc_label[1][0]
            sc_label_shape_c = sc_label[1][1]
            y_min, y_max = max(0.0, float(sc_label_shape_r[0] / image_height)), min(
                1.0, float(sc_label_shape_r[1] / image_height)
            )
            x_min, x_max = max(0.0, float(sc_label_shape_c[0] / image_width)), min(
                1.0, float(sc_label_shape_c[1] / image_width)
            )

            if sc_label_name == "ellipse":
                # Fix issue with newer scikit-image libraries that generate ellipses.
                # For now we render a rectangle on top of it
                sc_label_name = "rectangle"
                rr, cc = rectangle(
                    start=(sc_label_shape_r[0], sc_label_shape_c[0]),
                    end=(sc_label_shape_r[1] - 1, sc_label_shape_c[1] - 1),
                    shape=image1.shape,
                )
                image1[rr, cc] = (
                    random.randint(0, 200),
                    random.randint(0, 200),
                    random.randint(0, 200),
                )
            if sc_label_name == "circle":
                sc_label_name = "ellipse"

            label_matches = [label for label in labels if sc_label_name == label.name]
            if len(label_matches) > 0:
                label = label_matches[0]
                box_annotation = Annotation(
                    Box(x1=x_min, y1=y_min, x2=x_max, y2=y_max),
                    labels=[ScoredLabel(label, probability=1.0)],
                )

                annotation: Annotation

                if label.name == "ellipse":
                    annotation = Annotation(
                        Ellipse(
                            x1=box_annotation.shape.x1,
                            y1=box_annotation.shape.y1,
                            x2=box_annotation.shape.x2,
                            y2=box_annotation.shape.y2,
                        ),
                        labels=box_annotation.get_labels(include_empty=True),
                    )
                elif label.name == "triangle":
                    points = [
                        Point(
                            x=(box_annotation.shape.x1 + box_annotation.shape.x2) / 2,
                            y=box_annotation.shape.y1,
                        ),
                        Point(x=box_annotation.shape.x1, y=box_annotation.shape.y2),
                        Point(x=box_annotation.shape.x2, y=box_annotation.shape.y2),
                    ]

                    annotation = Annotation(
                        Polygon(points=points),
                        labels=box_annotation.get_labels(include_empty=True),
                    )
                else:
                    annotation = box_annotation

                annotations.append(annotation)
            else:
                logger.warning(
                    "Generated a random image, but was not able to associate a label with a shape. "
                    f"The name of the shape was `{sc_label_name}`. "
                )

        return image1, annotations

    def init_environment(self, params, model_template, number_of_images=500):
        labels = [
            Label(name='rectangle', domain="detection", id=0),
            Label(name='ellipse', domain="detection", id=1),
            Label(name='triangle', domain="detection", id=2)
        ]
        labels_schema = FlatLabelSchema(labels)
        labels_list = labels_schema.get_labels(False)
        environment = TaskEnvironment(model=None, hyper_parameters=params, label_schema=labels_schema,
                                      model_template=model_template)

        warnings.filterwarnings('ignore', message='.* coordinates .* are out of bounds.*')
        items = []
        for i in range(0, number_of_images):
            image_numpy, shapes = self.generate_random_annotated_image(image_width=640,
                                                                       image_height=480,
                                                                       labels=labels_list,
                                                                       max_shapes=20,
                                                                       min_size=50,
                                                                       max_size=100,
                                                                       random_seed=None)
            # Convert all shapes to bounding boxes
            box_shapes = []
            for shape in shapes:
                shape_labels = shape.get_labels(include_empty=True)
                shape = shape.shape
                if isinstance(shape, (Box, Ellipse)):
                    box = np.array([shape.x1, shape.y1, shape.x2, shape.y2], dtype=float)
                elif isinstance(shape, Polygon):
                    box = np.array([shape.min_x, shape.min_y, shape.max_x, shape.max_y], dtype=float)
                box = box.clip(0, 1)
                box_shapes.append(Annotation(Box(x1=box[0], y1=box[1], x2=box[2], y2=box[3]),
                                             labels=shape_labels))

            image = Image(data=image_numpy, name=f'image_{i}')
            items.append(MMDatasetItem(image, box_shapes))
        warnings.resetwarnings()

        rng = random.Random()
        rng.shuffle(items)
        for i, _ in enumerate(items):
            subset_region = i / number_of_images
            if subset_region >= 0.8:
                subset = Subset.TESTING
            elif subset_region >= 0.6:
                subset = Subset.VALIDATION
            else:
                subset = Subset.TRAINING
            items[i].subset = subset

        # dataset = Dataset(NullDatasetStorage(), items)
        dataset = MMDataset(items=items, labels=labels)
        return environment, dataset

    def setup_configurable_parameters(self, template_dir, num_iters=250):
        model_template = parse_model_template(osp.join(template_dir, 'template.yaml'), '1')

        # Here we have to reload parameters manually because
        # `parse_model_template` was called when `configuration.yaml` was not near `template.yaml.`
        if not model_template.hyper_parameters.data:
            reload_hyper_parameters(model_template)

        hyper_parameters = model_template.hyper_parameters.data
        set_values_as_default(hyper_parameters)
        hyper_parameters = create(hyper_parameters)
        hyper_parameters.learning_parameters.num_iters = num_iters
        hyper_parameters.learning_parameters.num_checkpoints = 1
        hyper_parameters.postprocessing.result_based_confidence_threshold = False
        hyper_parameters.postprocessing.confidence_threshold = 0.1
        return hyper_parameters, model_template

    def test_cancel_training_detection(self):
        """
        Tests starting and cancelling training.

        Flow of the test:
        - Creates a randomly annotated project with a small dataset containing 3 classes:
            ['rectangle', 'triangle', 'circle'].
        - Start training and give cancel training signal after 10 seconds. Assert that training
            stops within 35 seconds after that
        - Start training and give cancel signal immediately. Assert that training stops within 25 seconds.

        This test should be finished in under one minute on a workstation.
        """
        template_dir = osp.join('configs', 'ote', 'custom-object-detection', 'mobilenetV2_ATSS')
        hyper_parameters, model_template = self.setup_configurable_parameters(template_dir, num_iters=500)
        detection_environment, dataset = self.init_environment(hyper_parameters, model_template, 250)

        detection_task = OTEDetectionTask(task_environment=detection_environment)

        executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix='train_thread')

        output_model = Model(
                dataset,
                detection_environment.get_model_configuration(),
                model_status=ModelStatus.NOT_READY)

        # Test stopping after some time
        start_time = time.time()
        train_future = executor.submit(detection_task.train, dataset, output_model)
        # give train_thread some time to initialize the model
        while not detection_task._is_training:
            time.sleep(10)
        detection_task.cancel_training()

        # stopping process has to happen in less than 35 seconds
        train_future.result()
        self.assertLess(time.time() - start_time, 35, 'Expected to stop within 35 seconds.')

        # Test stopping immediately (as soon as training is started).
        start_time = time.time()
        train_future = executor.submit(detection_task.train, dataset, output_model)
        while not detection_task._is_training:
            time.sleep(0.1)
        detection_task.cancel_training()

        train_future.result()
        self.assertLess(time.time() - start_time, 25)  # stopping process has to happen in less than 25 seconds

    def test_training_progress_tracking(self):
        template_dir = osp.join('configs', 'ote', 'custom-object-detection', 'mobilenetV2_ATSS')
        hyper_parameters, model_template = self.setup_configurable_parameters(template_dir, num_iters=10)
        detection_environment, dataset = self.init_environment(hyper_parameters, model_template, 50)

        task = OTEDetectionTask(task_environment=detection_environment)
        self.addCleanup(task._delete_scratch_space)

        print('Task initialized, model training starts.')
        training_progress_curve = []

        def progress_callback(progress: float, score: Optional[float] = None):
            training_progress_curve.append(progress)

        train_parameters = TrainParameters
        train_parameters.update_progress = progress_callback
        output_model = Model(
                dataset,
                detection_environment.get_model_configuration(),
                model_status=ModelStatus.NOT_READY)
        task.train(dataset, output_model, train_parameters)

        self.assertGreater(len(training_progress_curve), 0)
        training_progress_curve = np.asarray(training_progress_curve)
        self.assertTrue(np.all(training_progress_curve[1:] >= training_progress_curve[:-1]))

    @staticmethod
    def eval(task: OTEDetectionTask, model: Model, dataset: MMDataset) -> Performance:
        start_time = time.time()
        result_dataset = task.infer(dataset.with_empty_annotations())
        end_time = time.time()
        print(f'{len(dataset)} analysed in {end_time - start_time} seconds')
        result_set = ResultSet(
            model=model,
            ground_truth_dataset=dataset,
            prediction_dataset=result_dataset
        )
        performance = task.evaluate(result_set)
        return performance

    def train_and_eval(self, template_dir):
        """
        Run training, analysis, evaluation and model optimization

        Flow of the test:
        - Creates a randomly annotated project with a small dataset containing 3 classes:
            ['rectangle', 'triangle', 'circle'].
        - Trains a model for 10 epochs. Asserts that validation F-measure is larger than the threshold and
            also that OpenVINO optimization runs successfully.
        - Reloads the model in the task and recompute the performance. Asserts that the performance
            difference between the original and the reloaded model is smaller than 1e-4. Ideally there should be no
            difference at all.
        """
        hyper_parameters, model_template = self.setup_configurable_parameters(template_dir, num_iters=150)
        detection_environment, dataset = self.init_environment(hyper_parameters, model_template, 250)

        val_dataset = dataset.get_subset(Subset.VALIDATION)
        task = OTEDetectionTask(task_environment=detection_environment)
        self.addCleanup(task._delete_scratch_space)

        print('Task initialized, model training starts.')
        # Train the task.
        # train_task checks that the task returns an OptimizedModel and that
        # validation f-measure is higher than the threshold, which is a pretty low bar
        # considering that the dataset is so easy
        output_model = Model(
                dataset,
                detection_environment.get_model_configuration(),
                model_status=ModelStatus.NOT_READY)
        task.train(dataset, output_model)

        # Test that labels and configurable parameters are stored in model.data
        modelinfo = torch.load(io.BytesIO(output_model.get_data("weights.pth")))
        self.assertEqual(list(modelinfo.keys()), ['model', 'config', 'labels', 'VERSION'])
        self.assertTrue('ellipse' in modelinfo['labels'])

        if isinstance(task, IExportTask):
            exported_model = OptimizedModel(
                NullProject(),
                NullModelStorage(),
                dataset,
                detection_environment.get_model_configuration(),
                ModelOptimizationType.MO,
                precision=[ModelPrecision.FP32],
                optimization_methods=[],
                optimization_level={},
                target_device=TargetDevice.UNSPECIFIED,
                performance_improvement={},
                model_size_reduction=1.,
                model_status=ModelStatus.NOT_READY)
            task.export(ExportType.OPENVINO, exported_model)

        # Run inference
        validation_performance = self.eval(task, output_model, val_dataset)
        print(f'Evaluated model to have a performance of {validation_performance}')
        score_threshold = 0.5
        self.assertGreater(validation_performance.score.value, score_threshold,
            f'Expected F-measure to be higher than {score_threshold}')

        print('Reloading model.')
        first_model = output_model
        new_model = Model(
            dataset,
            detection_environment.get_model_configuration(),
            model_status=ModelStatus.NOT_READY)
        task._hyperparams.learning_parameters.num_iters = 10
        task._hyperparams.learning_parameters.num_checkpoints = 1
        task.train(dataset, new_model)
        self.assertTrue(first_model.model_status)
        self.assertNotEqual(first_model, new_model)

        # Make the new model fail
        new_model.model_status = ModelStatus.NOT_IMPROVED
        detection_environment.model = first_model
        task = OTEDetectionTask(detection_environment)
        self.assertEqual(task._task_environment.model.id, first_model.id)

        print('Reevaluating model.')
        # Performance should be the same after reloading
        performance_after_reloading = self.eval(task, output_model, val_dataset)
        performance_delta = performance_after_reloading.score.value - validation_performance.score.value
        perf_delta_tolerance = 0.0

        self.assertEqual(np.abs(performance_delta), perf_delta_tolerance,
                         msg=f'Expected no performance difference after reloading. Performance delta '
                             f'({validation_performance.score.value} vs {performance_after_reloading.score.value}) was '
                             f'larger than the tolerance of {perf_delta_tolerance}')

        print(f'Performance: {validation_performance.score.value:.4f}')
        print(f'Performance after reloading: {performance_after_reloading.score.value:.4f}')
        print(f'Performance delta after reloading: {performance_delta:.6f}')

        if isinstance(task, IExportTask):
            detection_environment.model = exported_model
            ov_task = OpenVINODetectionTask(detection_environment)
            predicted_validation_dataset = ov_task.infer(val_dataset.with_empty_annotations())
            resultset = ResultSet(
                model=output_model,
                ground_truth_dataset=val_dataset,
                prediction_dataset=predicted_validation_dataset,
            )
            export_performance = ov_task.evaluate(resultset)
            print(export_performance)
            performance_delta = export_performance.score.value - validation_performance.score.value
            perf_delta_tolerance = 0.005
            self.assertLess(np.abs(performance_delta), perf_delta_tolerance,
                        msg=f'Expected no or very small performance difference after export. Performance delta '
                            f'({validation_performance.score.value} vs {export_performance.score.value}) was '
                            f'larger than the tolerance of {perf_delta_tolerance}')

    def test_training_custom_mobilenetssd_256(self):
        self.train_and_eval(osp.join('configs', 'ote', 'custom-object-detection', 'mobilenet_v2-2s_ssd-256x256'))

    def test_training_custom_mobilenetssd_384(self):
        self.train_and_eval(osp.join('configs', 'ote', 'custom-object-detection', 'mobilenet_v2-2s_ssd-384x384'))

    def test_training_custom_mobilenetssd_512(self):
        self.train_and_eval(osp.join('configs', 'ote', 'custom-object-detection', 'mobilenet_v2-2s_ssd-512x512'))

    def test_training_custom_mobilenet_atss(self):
        self.train_and_eval(osp.join('configs', 'ote', 'custom-object-detection', 'mobilenetV2_ATSS'))

    def test_training_custom_mobilenet_ssd(self):
        self.train_and_eval(osp.join('configs', 'ote', 'custom-object-detection', 'mobilenetV2_SSD'))

    def test_training_custom_resnet_vfnet(self):
        self.train_and_eval(osp.join('configs', 'ote', 'custom-object-detection', 'resnet50_VFNet'))
